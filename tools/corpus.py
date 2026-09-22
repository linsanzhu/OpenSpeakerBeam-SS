"""
Corpus indexers used by the mixture builder and the training data loader.

Each indexer walks a dataset's directory layout and returns a
`{speaker_id: [wav_path, ...]}` mapping (or just a flat list for noise /
RIR collections). Indexes are pickled under `data/indexes/` so a 360-hour
LibriSpeech + 2300-hour VoxCeleb2 scan takes seconds on subsequent runs
instead of minutes.
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass


@dataclass
class Index:
    """Pickle-able handle to a cached corpus scan."""
    path: str  # absolute path to the pickle

    def exists(self) -> bool:
        return os.path.isfile(self.path)


def _cache_path(name: str) -> str:
    """Resolve the pickle path for a named corpus index."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "data", "indexes"), exist_ok=True)
    return os.path.join(here, "data", "indexes", f"{name}.pkl")


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------
def index_librispeech(split_dir: str, cache_name: str) -> dict[str, list[str]]:
    """Walk `<root>/<speaker>/<chapter>/<file>.flac` into speaker -> files."""
    cache = _cache_path(cache_name)
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    speakers: dict[str, list[str]] = {}
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")
    for speaker in sorted(os.listdir(split_dir)):
        spk_dir = os.path.join(split_dir, speaker)
        if not os.path.isdir(spk_dir):
            continue
        files = sorted(glob.glob(os.path.join(spk_dir, "*", "*.flac")))
        if files:
            speakers[speaker] = files
    with open(cache, "wb") as f:
        pickle.dump(speakers, f)
    return speakers


# ---------------------------------------------------------------------------
# VoxCeleb2 (id-style: <id>/<video>/<chunk>.wav, no real speaker identity)
# ---------------------------------------------------------------------------
def index_voxceleb2(dev_dir: str, cache_name: str = "voxceleb2") -> dict[str, list[str]]:
    """Walk VoxCeleb2 dev: each subdir under `dev_dir` is one identity."""
    cache = _cache_path(cache_name)
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    if not os.path.isdir(dev_dir):
        raise FileNotFoundError(f"VoxCeleb2 dev not found: {dev_dir}")
    wav_root = os.path.join(dev_dir, "wav")
    if not os.path.isdir(wav_root):
        # VoxCeleb2 mirrors sometimes ship a flat layout. Fall back to dev_dir.
        wav_root = dev_dir
    speakers: dict[str, list[str]] = {}
    for identity in sorted(os.listdir(wav_root)):
        id_dir = os.path.join(wav_root, identity)
        if not os.path.isdir(id_dir):
            continue
        files = sorted(glob.glob(os.path.join(id_dir, "*", "*.wav")))
        if files:
            speakers[identity] = files
    with open(cache, "wb") as f:
        pickle.dump(speakers, f)
    return speakers


# ---------------------------------------------------------------------------
# Noise (WHAM! layout: <root>/{tr,cv,tt}/*.wav)
# ---------------------------------------------------------------------------
def index_noise(root: str, cache_name: str) -> list[str]:
    """Flat list of noise wav paths."""
    cache = _cache_path(cache_name)
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Noise root not found: {root}")
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    with open(cache, "wb") as f:
        pickle.dump(files, f)
    return files


# ---------------------------------------------------------------------------
# RIR (WHAMR! layout: <root>/rirs_*.wav, sometimes split per condition)
# ---------------------------------------------------------------------------
def index_rirs(root: str, cache_name: str) -> list[str]:
    """Flat list of room impulse response paths."""
    cache = _cache_path(cache_name)
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"RIR root not found: {root}")
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    with open(cache, "wb") as f:
        pickle.dump(files, f)
    return files
