"""Build an inspection sheet from existing evidence, without inventing labels."""

import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    events_path = Path("outputs/validation/v3_1/baseline_events.json")
    events = json.loads(events_path.read_text(encoding="utf-8"))["events"]
    tiles = []
    for event in events:
        image = cv2.imread(str(events_path.parent/event["evidence_path"]))
        x1,y1,x2,y2 = map(round,event["bbox"])
        crop = image[max(0,y1-150):min(image.shape[0],y2+150),max(0,x1-150):min(image.shape[1],x2+150)]
        scale = min(600/crop.shape[1],280/crop.shape[0])
        crop = cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
        tile = np.full((320,620,3),24,np.uint8)
        tile[35:35+crop.shape[0],10:10+crop.shape[1]] = crop
        cv2.putText(tile,f"EVENT {event['event_id']} / F{event['frame']} / {event['confidence']:.0%}",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
        tiles.append(tile)
    for group in range(0,len(tiles),6):
        subset = tiles[group:group+6]
        while len(subset)%2:
            subset.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(subset[i:i+2]) for i in range(0,len(subset),2)])
        cv2.imwrite(str(events_path.parent/f"potholes_review_{group//6+1}.jpg"),sheet)


if __name__ == "__main__":
    main()
