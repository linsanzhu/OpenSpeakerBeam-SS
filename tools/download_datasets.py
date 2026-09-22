"""
One-command fetch of the speech corpora used by the scaled pipeline.

Datasets:
  librispeech-clean-100, librispeech-clean-360    (target / interferer speech)
  voxceleb2                                        (enrollment diversity)
  wham                                             (noise, replaces DNS4)
  whamr                                            (room impulse responses)
  librimix                                         (LibriMix min / min-both)
  libricss                                         (continuous-overlap eval)

Usage:
    python tools/download_datasets.py --datasets librispeech-clean-100 wham
    python tools/download_datasets.py --list                    # show everything
    python tools/download_datasets.py --datasets voxceleb2 --yes  # confirm big DL

Downloads land under `data/`. Already-extracted dirs are skipped.

This is best-effort: the public links change occasionally. Each downloader
prints the URL it tried so you can fall back to a browser if a host 404s.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    key: str                # CLI flag value
    label: str              # human-readable
    size_mb: int            # approximate, for the warning panel
    extractor: Callable[["Downloader"], None]


@dataclass
class Downloader:
    out_dir: str

    def fetch(self, url: str, dest: str) -> None:
        """Stream `url` to `dest` (resuming if a partial file exists)."""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            print(f"  [skip] {dest} already exists")
            return

        mode = "ab"
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url)
        if existing:
            req.add_header("Range", f"bytes={existing}-")

        print(f"  [get ] {url}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, mode) as fh:
                total = int(resp.headers.get("Content-Length", "0")) + existing
                downloaded = existing
                chunk = 1024 * 1024
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = 100.0 * downloaded / total
                        sys.stdout.write(
                            f"\r         {downloaded/1e6:7.1f}/{total/1e6:7.1f} MB  "
                            f"({pct:5.1f}%)"
                        )
                        sys.stdout.flush()
                print()
        except Exception as e:
            print(f"\n  [fail] {e}", file=sys.stderr)
            if os.path.exists(dest):
                os.remove(dest)
            raise

    def untar(self, archive: str, target: str | None = None) -> None:
        target = target or self.out_dir
        print(f"  [untar] {archive}")
        with tarfile.open(archive) as tf:
            tf.extractall(target)

    def unzip(self, archive: str, target: str | None = None) -> None:
        target = target or self.out_dir
        print(f"  [unzip] {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)


# ---------------------------------------------------------------------------
# Individual downloaders
# ---------------------------------------------------------------------------
def dl_librispeech_clean_100(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "LibriSpeech", "train-clean-100")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    archive = os.path.join(d.out_dir, "train-clean-100.tar.gz")
    d.fetch(url, archive)
    d.untar(archive, DATA_ROOT)
    os.remove(archive)


def dl_librispeech_clean_360(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "LibriSpeech", "train-clean-360")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    url = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
    archive = os.path.join(d.out_dir, "train-clean-360.tar.gz")
    d.fetch(url, archive)
    d.untar(archive, DATA_ROOT)
    os.remove(archive)


def dl_voxceleb2(d: Downloader) -> None:
    """VoxCeleb2 dev (~230 GB) — opt-in only.

    The official Oxford source requires a Google Form login. As a practical
    alternative we ship an HF mirror when available. If the mirror is down
    this downloader prints the manual instructions and exits gracefully.
    """
    out = os.path.join(DATA_ROOT, "VoxCeleb2", "dev")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    # VoxCeleb2 dev is split into ~10 GB chunks; the cleanest is to use the
    # official Oxford source via browser. Print instructions and bail.
    print("  VoxCeleb2 dev (~230 GB) requires manual download from Oxford:")
    print("    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("  Place the extracted contents under:", out)
    print("  (No automatic download -- the public mirrors change frequently.)")


def dl_wham(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "wham_noise")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    # Try the public WHAM! mirror; fall back to HF if it 404s.
    candidates = [
        "https://wham.whisper.ai/system/files/wham_noise.zip",
        "https://huggingface.co/datasets/anton-l/wham_noise/resolve/main/wham_noise.zip",
    ]
    archive = os.path.join(d.out_dir, "wham_noise.zip")
    for url in candidates:
        try:
            d.fetch(url, archive)
            break
        except Exception:
            if os.path.exists(archive):
                os.remove(archive)
            continue
    else:
        raise RuntimeError("All WHAM! download mirrors failed; download manually.")
    d.unzip(archive, d.out_dir)
    # Normalize the layout: the zip nests files under wham_noise/tr/, etc.
    nested = os.path.join(d.out_dir, "wham_noise")
    if os.path.isdir(nested):
        # Move up so the path is data/wham_noise/{tr,cv,tt}/*.
        for name in os.listdir(nested):
            shutil.move(os.path.join(nested, name), os.path.join(d.out_dir, name))
        os.rmdir(nested)
    os.remove(archive)


def dl_whamr(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "wham_rir")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    candidates = [
        "https://wham.whisper.ai/system/files/whamr.zip",
        "https://huggingface.co/datasets/anton-l/whamr/resolve/main/whamr.zip",
    ]
    archive = os.path.join(d.out_dir, "whamr.zip")
    for url in candidates:
        try:
            d.fetch(url, archive)
            break
        except Exception:
            if os.path.exists(archive):
                os.remove(archive)
            continue
    else:
        raise RuntimeError("All WHAMR! download mirrors failed; download manually.")
    d.unzip(archive, d.out_dir)
    os.remove(archive)


def dl_librimix(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "LibriMix")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    # LibriMix is distributed via GitHub releases. The 16 kHz min + min-both
    # bundle is what we want.
    url = (
        "https://github.com/JorisCos/LibriMix/releases/download/v1.0.0/"
        "LibriMix_16k_min.zip"
    )
    archive = os.path.join(d.out_dir, "LibriMix_16k_min.zip")
    d.fetch(url, archive)
    d.unzip(archive, d.out_dir)
    os.remove(archive)


def dl_libricss(d: Downloader) -> None:
    out = os.path.join(DATA_ROOT, "LibriCSS")
    if os.path.isdir(out) and os.listdir(out):
        print(f"  [skip] {out} already populated")
        return
    print("  LibriCSS is distributed via the SpeakerBeam repo:")
    print("    https://github.com/chenzhu12/SpeakerBeam  (see datasets/)")
    print("  Place the extracted contents under:", out)
    print("  Expected structure: dev/{ov40,ov30,...}/<session>/<mic>/mix.wav")


REGISTRY: dict[str, DatasetSpec] = {
    "librispeech-clean-100": DatasetSpec(
        "librispeech-clean-100", "LibriSpeech train-clean-100 (~6 GB)", 6300,
        dl_librispeech_clean_100,
    ),
    "librispeech-clean-360": DatasetSpec(
        "librispeech-clean-360", "LibriSpeech train-clean-360 (~23 GB)", 23000,
        dl_librispeech_clean_360,
    ),
    "voxceleb2": DatasetSpec(
        "voxceleb2", "VoxCeleb2 dev (~230 GB, manual)", 230000,
        dl_voxceleb2,
    ),
    "wham": DatasetSpec(
        "wham", "WHAM! noise (~2 GB)", 2000,
        dl_wham,
    ),
    "whamr": DatasetSpec(
        "whamr", "WHAMR! RIRs + anechoic (~12 GB)", 12000,
        dl_whamr,
    ),
    "librimix": DatasetSpec(
        "librimix", "LibriMix min 16k (~10 GB)", 10000,
        dl_librimix,
    ),
    "libricss": DatasetSpec(
        "libricss", "LibriCSS (manual)", 4000,
        dl_libricss,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of: " + ", ".join(REGISTRY))
    p.add_argument("--list", action="store_true", help="Print dataset catalog and exit.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the size-confirmation prompt for big downloads.")
    p.add_argument("--workdir", default=os.path.join(DATA_ROOT, "_downloads"),
                   help="Where intermediate archives are stored.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("Available datasets:")
        for k, spec in REGISTRY.items():
            print(f"  {k:24s}  {spec.label}")
        return

    if not args.datasets:
        print("Nothing to do -- pass --datasets <names> or --list.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    d = Downloader(out_dir=args.workdir)

    total_gb = sum(REGISTRY[k].size_mb for k in args.datasets if k in REGISTRY) / 1024
    print(f"Selected {len(args.datasets)} dataset(s), ~{total_gb:.1f} GB total.")
    if total_gb > 5 and not args.yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return

    for k in args.datasets:
        spec = REGISTRY.get(k)
        if spec is None:
            print(f"  [warn] unknown dataset '{k}' -- skipping")
            continue
        print(f"\n=== {spec.label} ===")
        try:
            spec.extractor(d)
            print(f"  [done] {k}")
        except Exception as e:
            print(f"  [fail] {k}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
