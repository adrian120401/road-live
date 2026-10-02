"""Extract one local visual prompt; no training or network inference."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from download_crosswalk_model import URL, SHA256, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("video2.MOV"))
    parser.add_argument("--frame",type=int,default=270)
    parser.add_argument("--box",type=float,nargs=4,default=(12,2180,2155,2510))
    parser.add_argument("--reference",type=Path,help="Imagen local alternativa al frame del video")
    parser.add_argument("--device",default="cpu",help="Dispositivo para preparar el ejemplo; no cambia inferencia")
    args = parser.parse_args()
    if args.reference:
        frame = cv2.imread(str(args.reference))
        if frame is None:
            raise ValueError("Cannot read visual reference image")
    else:
        capture = cv2.VideoCapture(str(args.input))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES,args.frame-1)
            ok,frame = capture.read()
            if not ok:
                raise ValueError("Cannot read visual reference")
        finally:
            capture.release()
    weights = Path("models/yoloe-26s-seg.pt")
    verify(weights)
    model = YOLOE(str(weights))
    model.predict(frame,refer_image=frame,visual_prompts={"bboxes":np.array([args.box]),"cls":np.array([0])},
                  predictor=YOLOEVPSegPredictor,imgsz=640,device=args.device,verbose=False,save=False)
    prompts = weights.with_name("yoloe-26s-crosswalk-visual.npz")
    model.save_prompt_embeddings(prompts)
    reference = weights.with_name("crosswalk_reference.jpg")
    cv2.imwrite(str(reference),frame)
    manifest_path = weights.with_suffix(".json")
    previous = {"source":URL,"sha256":SHA256,"architecture":"YOLOE-26s-seg","input_size":640,
                "license":"AGPL-3.0 / Ultralytics Enterprise"}
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("prompt_mode") != "visual":
            weights.with_name("yoloe-26s-text-review.json").write_text(json.dumps(old,indent=2))
    with prompts.open("rb") as stream:
        sha = hashlib.file_digest(stream,"sha256").hexdigest()
    manifest = {**previous,"names":{0:"object0"},"crosswalk_class":"object0",
                "prompt_embeddings":prompts.name,"prompt_sha256":sha,"prompt_mode":"visual",
                "reference":{"input":str(args.reference or args.input),"frame":args.frame if not args.reference else None,
                             "bbox":args.box,"image":reference.name},
                "limitations":"One-shot visual prompt from this video, no training. Same-scene validation does not measure generalization."}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Prepared local visual reference:",prompts,flush=True)


if __name__ == "__main__":
    main()
