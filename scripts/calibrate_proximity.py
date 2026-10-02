"""Camera-specific profile from manually reviewed SAFE/CAUTION/NEAR frame groups."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.config import Config, Detection, ProximityConfig
from src.depth import DepthEstimator
from src.proximity import in_corridor
from src.tracker import ObjectTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("video2.MOV"))
    parser.add_argument("--output",type=Path,default=Path("assets/proximity_video2.json"))
    parser.add_argument("--safe",nargs=2,type=int,default=(950,1000))
    parser.add_argument("--caution",nargs=2,type=int,default=(1050,1110))
    parser.add_argument("--near",nargs=2,type=int,default=(1170,1250))
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS)
    general = ObjectTracker(Config(args.input))
    config = ProximityConfig(enabled=True)
    depth = DepthEstimator(config,general.device,fps)
    groups = {}
    try:
        for label,bounds in (("SAFE",args.safe),("CAUTION",args.caution),("NEAR",args.near)):
            rows = []
            for number in range(bounds[0],bounds[1]+1,5):
                capture.set(cv2.CAP_PROP_POS_FRAMES,number-1)
                ok,frame = capture.read()
                if not ok:
                    raise ValueError(f"Invalid reviewed frame: {number}")
                result = next(general.model.predict(frame,imgsz=640,device=general.device,conf=.1,
                                                    classes=general.class_ids,stream=True,verbose=False))
                detections = [Detection(i+1,general.model.names[int(c)],float(s),tuple(map(float,b)))
                              for i,(b,c,s) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()))]
                observations,_ = depth.update(frame,detections,number,(number-1)/fps,has_candidate=True)
                candidates = [observations[d.track_id] for d in detections if d.track_id in observations
                              and in_corridor(d,config,frame.shape[:2])]
                if candidates:
                    chosen = min(candidates,key=lambda o:o.estimate)
                    rows.append({"frame":number,**asdict(chosen)})
            if len(rows) < 5:
                raise ValueError(f"Not enough valid samples in manually reviewed {label} group")
            groups[label] = {"frames":list(bounds),"median":float(np.median([r["estimate"] for r in rows])),
                             "observations":rows}
            print(label,groups[label]["median"],"samples",len(rows),flush=True)
    finally:
        capture.release()
    near, caution, safe = (groups[k]["median"] for k in ("NEAR","CAUTION","SAFE"))
    if not 0 < near < caution < safe:
        raise ValueError("Reviewed depth groups are not ordered; do not enable warnings")
    near_threshold,caution_threshold = float(np.sqrt(near*caution)),float(np.sqrt(caution*safe))
    # Frame-to-frame noise within reviewed groups, capped to preserve separation.
    deltas = [abs(a["estimate"]-b["estimate"]) for g in groups.values()
              for a,b in zip(g["observations"],g["observations"][1:])]
    hysteresis = min(float(np.percentile(deltas,90)),(caution_threshold-near_threshold)/4)
    profile = {"near_threshold":near_threshold,"caution_threshold":caution_threshold,"hysteresis":hysteresis,
               "input":str(args.input),"depth_model":str(config.depth_model),"depth_image_size":config.depth_image_size,
               "corridor":config.corridor,"groups":groups,"metric_calibrated":False,
               "candidate_confidence":config.candidate_confidence,
               "method":"Geometric midpoints of reviewed group medians; P90 temporal deltas for hysteresis",
               "review":"Forward motorcycle and pickup approaching intersection; labels are qualitative, not safety distances",
               "limitations":"Specific to reviewed camera/video; recalibrate for changed mounting/viewpoint."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2),encoding="utf-8")
    print("Profile:",args.output,near_threshold,caution_threshold,hysteresis,flush=True)


if __name__ == "__main__":
    main()
