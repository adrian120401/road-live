"""Diagnostic prompt/crop evaluation on the reviewed crossing and controls."""

import json
from pathlib import Path

import cv2
from ultralytics import YOLOE


def main():
    model = YOLOE("models/yoloe-26s-seg.pt")
    model.load_prompt_embeddings("models/yoloe-26s-seg.npz")
    capture = cv2.VideoCapture("video2.MOV")
    rows = []
    try:
        for number in (240,270,285,300,315,1191):
            capture.set(cv2.CAP_PROP_POS_FRAMES,number-1)
            _,frame = capture.read()
            h = frame.shape[0]
            for mode,roi in (("full",frame),("road",frame[round(.42*h):round(.74*h)])):
                result = next(model.predict(roi,imgsz=640,device=0,conf=.01,stream=True,verbose=False))
                detections = [{"class":result.names[int(c)],"confidence":float(s),"box":list(map(float,b))}
                              for b,c,s in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy())]
                rows.append({"frame":number,"crop":mode,"detections":detections})
                print(number,mode,[(d['class'],round(d['confidence'],3),[round(v) for v in d['box']]) for d in detections[:5]],flush=True)
    finally:
        capture.release()
    Path("outputs/validation/v3_1/crosswalk_prompt_review.json").write_text(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
