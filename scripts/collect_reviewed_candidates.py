"""Apply explicit visual-review labels; no automatic labels or training."""

import argparse
import json
from pathlib import Path
import shutil

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events",type=Path,required=True)
    parser.add_argument("--review",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("dataset_candidates/potholes"))
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    events = {e["event_id"]:e for e in json.loads(args.events.read_text(encoding="utf-8"))["events"]}
    for label in ("true_positive","false_positive","false_negative","unreviewed"):
        (args.output/label).mkdir(parents=True,exist_ok=True)
    for row in review["events"]:
        event = events[row["event_id"]]
        label = row["label"]
        if label not in {"true_positive","false_positive","unreviewed"}:
            raise ValueError("Invalid review category")
        source = event.get("candidate_path")
        if not source or not Path(source).is_file():
            raise ValueError("Run with --collect-road-candidates to preserve unannotated frames")
        path = args.output/label/f"{args.events.stem}_event_{event['event_id']:03d}.jpg"
        shutil.copy2(source,path)
        path.with_suffix(".json").write_text(json.dumps({"event":event,"review":row,
                "review_status":"provisional visual review; requires human dataset review"},indent=2),encoding="utf-8")
    capture = cv2.VideoCapture(str(args.input))
    try:
        for row in review.get("false_negative_candidates",[]):
            number = row["frame"]
            capture.set(cv2.CAP_PROP_POS_FRAMES,number-1)
            ok,image = capture.read()
            if not ok:
                raise ValueError("Cannot decode candidate frame")
            path = args.output/"false_negative"/f"{args.input.stem}_frame_{number:04d}.jpg"
            if not cv2.imwrite(str(path),image):
                raise OSError("Cannot save candidate image")
            path.with_suffix(".json").write_text(json.dumps({**row,"input":str(args.input),
                "review_status":"candidate; temporal miss, not a certified missed physical event"},indent=2),encoding="utf-8")
    finally:
        capture.release()
    print("Review images collected:",args.output,flush=True)


if __name__ == "__main__":
    main()
