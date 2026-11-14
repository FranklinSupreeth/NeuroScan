"""
Make sure the trained model files are present before the app starts.

The main model (Best_BrainTumor_Model.pkl, ~257 MB) is larger than GitHub's
100 MB file limit, so it is usually stored with Git LFS. If Render clones the
repo without the LFS objects, the .pkl files arrive as tiny text "pointer"
files. This script detects that and downloads the real files.

Runs automatically in the Render build command. Safe to run repeatedly.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

# Where to fetch the files from if they are missing. Override with the
# MODEL_BASE_URL env var if you host the models somewhere else
# (e.g. your own GitHub repo's LFS media URL, S3, Google Drive direct link).
DEFAULT_BASE_URL = (
    "https://media.githubusercontent.com/media/"
    "jishnu395/Brain_Tumor_Model/main/saved_models"
)
BASE_URL = os.environ.get("MODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

FILES = {
    "Best_BrainTumor_Model.pkl": "b898ba4c6001f39078be4f1e092648e908210d76cc77f7f8f79a7dfa8baa81dd",
    "Scaler.pkl": None,
    "LabelEncoder.pkl": None,
}


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest: Path) -> None:
    url = f"{BASE_URL}/{name}"
    tmp = dest.with_suffix(".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def main() -> int:
    MODEL_DIR.mkdir(exist_ok=True)
    ok = True

    for name, expected_hash in FILES.items():
        path = MODEL_DIR / name
        needs = (not path.exists()) or is_lfs_pointer(path)

        if not needs:
            print(f"[ok] {name} ({path.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"[missing] {name}")
        try:
            download(name, path)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: could not download {name}: {e}")
            ok = False
            continue

        if expected_hash and sha256(path) != expected_hash:
            print(f"  WARNING: checksum mismatch for {name}")
        print(f"[ok] {name} ({path.stat().st_size / 1e6:.1f} MB)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
