"""Review source frames and depth distributions; does not change event outputs."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.config import Config, CrosswalkConfig, Detection, ProximityConfig
from src.crosswalk import CrosswalkDetector
from src.depth import DepthEstimator
from src.proximity import in_corridor
from src.tracker import ObjectTracker


def main() -> None:
    root = Path("outputs/validation/v3_1/probe")
    root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture("video2.MOV")
    fps = capture.get(cv2.CAP_PROP_FPS)
    general = ObjectTracker(Config(Path("video2.MOV")))
    proximity = ProximityConfig(enabled=True)
    depth = DepthEstimator(proximity,general.device,fps)
    crossing = None
    if CrosswalkConfig().model.with_suffix(".json").is_file():
        crossing = CrosswalkDetector(CrosswalkConfig(enabled=True),fps,2160,3840,general.device)
    rows = []
    try:
        for number in (1,180,240,270,300,600,900,960,1000,1033,1070,1110,1150,1191,1230,1270,1350,1450,1550,1650,1800):
            capture.set(cv2.CAP_PROP_POS_FRAMES,number-1)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {number}")
            # Independent images: IDs below only identify boxes in this diagnostic.
            result = next(general.model.predict(frame, imgsz=640,device=general.device,classes=general.class_ids,
                                                conf=.10,stream=True,verbose=False))
            detections = [Detection(i+1,general.model.names[int(c)],float(s),tuple(map(float,b)))
                          for i,(b,c,s) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),
                                                         result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy()))]
            observed,fresh = depth.update(frame,detections,number,(number-1)/fps,has_candidate=True)
            candidates = [asdict(observed[d.track_id]) for d in detections if d.track_id in observed
                          and in_corridor(d,proximity,frame.shape[:2])]
            crosses = crossing._predict(frame,record=True) if crossing else ()
            rows.append({"frame":number,"candidates":candidates,
                         "crossings":[asdict(d) for d in crosses],
                         "vehicle_observations":[{**asdict(o),"class":next(d.class_name for d in detections if d.track_id==i)}
                                                 for i,o in observed.items()]})
            working = cv2.resize(frame,(540,960))
            polygon = np.asarray([(int(x*540),int(y*960)) for x,y in proximity.corridor],np.int32)
            cv2.polylines(working,[polygon],True,(160,190,190),1)
            for d in detections:
                if d.track_id not in observed:
                    continue
                x1,y1,x2,y2 = (round(v/4) for v in d.box)
                candidate = in_corridor(d,proximity,frame.shape[:2])
                color = (0,220,255) if candidate else (170,170,170)
                cv2.rectangle(working,(x1,y1),(x2,y2),color,1)
                cv2.putText(working,f"{d.class_name} {observed[d.track_id].estimate:.2f}",(x1,max(12,y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
            for d in crosses:
                x1,y1,x2,y2 = (round(v/4) for v in d.box)
                cv2.rectangle(working,(x1,y1),(x2,y2),(255,210,100),2)
                cv2.putText(working,f"CROSSING {d.confidence:.2f}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,210,100),1)
            cv2.putText(working,f"FRAME {number}",(15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.imwrite(str(root/f"frame_{number:04d}.jpg"),working)
            print(number,[(c['track_id'],round(c['estimate'],2)) for c in candidates],
                  "crossings",[round(d.confidence,2) for d in crosses],flush=True)
    finally:
        capture.release()
    (root/"samples.json").write_text(json.dumps({"samples":rows,"depth":depth.report(),
                                                "crosswalk":crossing.report() if crossing else None},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
