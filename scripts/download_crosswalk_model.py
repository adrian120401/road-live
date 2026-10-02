"""Download and verify the official YOLOE crosswalk backbone."""

import hashlib
from pathlib import Path
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt"
SHA256 = "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654"
ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path) -> None:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != SHA256:
            raise ValueError(f"Official crosswalk backbone checksum mismatch: {path}")


def main() -> None:
    path = ROOT/"models/yoloe-26s-seg.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(URL,temporary)
        verify(temporary)
        temporary.replace(path)
    verify(path)
    print("Verified official local backbone:",path)


if __name__ == "__main__":
    main()
