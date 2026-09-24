import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("ealaxi/paysim1", "paysim1"),
    ("ieee-fraud-detection", "ieee-fraud-detection"),
]

OFAC_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_kaggle(ref: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    if "/" in ref:
        run([
            "kaggle", "datasets", "download",
            "-d", ref,
            "-p", str(out_dir)
        ])
    else:
        run([
            "kaggle", "competitions", "download",
            "-c", ref,
            "-p", str(out_dir)
        ])

    for z in out_dir.glob("*.zip"):
        print("extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(out_dir)
        z.unlink()


def download_direct(url: str, dest: Path):
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def main():
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("WARNING: KAGGLE_USERNAME/KAGGLE_KEY not set — skipping Kaggle downloads")
    else:
        for ref, name in DATASETS:
            download_kaggle(ref, RAW_DIR / name)

    download_direct(OFAC_URL, RAW_DIR / "sdn.csv")

    print("done. Next: dvc add data/raw && dvc push")


if __name__ == "__main__":
    main()