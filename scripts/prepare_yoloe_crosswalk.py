"""Prepare official text prompts once; video inference needs no text encoder/network."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools/yoloe-prepare-deps"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main() -> None:
    from ultralytics import YOLOE
    model_path = ROOT/"models/yoloe-26s-seg.pt"
    model = YOLOE(str(model_path))
    classes = ["crosswalk", "pedestrian crossing", "zebra crossing"]
    model.set_classes(classes)
    prompts = model_path.with_suffix(".npz")
    model.save_prompt_embeddings(prompts)
    manifest = {"source":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt",
                "sha256":digest(model_path),"architecture":"YOLOE-26s-seg",
                "input_size":640,"names":dict(enumerate(classes)),"crosswalk_class":classes[0],
                "prompt_embeddings":prompts.name,"prompt_sha256":digest(prompts),
                "text_encoder_sha256":digest(ROOT/"mobileclip2_b.ts"),
                "clip_revision":"a13192f8cb767260d7dfd98c843b0716593169e7",
                "license":"AGPL-3.0 / Ultralytics Enterprise",
                "limitations":"Open vocabulary; not a locally trained crosswalk model. Validate local viewpoint."}
    model_path.with_suffix(".json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Prepared offline crosswalk prompts:",prompts,flush=True)


if __name__ == "__main__":
    main()
