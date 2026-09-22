"""
Generate curriculum-style synthetic mixtures for SpeakerBeam-SS training.

Replaces the original flat 50k-mixture generator with:

  * Multi-source enrollment (VoxCeleb2 + LibriSpeech) to break the leak
    from "same speaker's own file" enrollment.
  * WHAM! noise (or the legacy `data/noise_fullband/` DNS4 fallback).
  * Optional WHAMR! reverberation toggle.
  * Curriculum phases: easy SIR/SNR first, hard later. Each phase writes
    a separate CSV so training can advance phase-by-phase.

Default layout:
    data_csv/<phase>/metadata.csv   with columns
        mixture_path, enrollment_path, target_path, sir_db, snr_db

CLI:
    python create_mixture_data_and_csv.py --config configs/curriculum.yaml
    python create_mixture_data_and_csv.py --num-mixtures 200000 --phase-easy 0.4 \
        --phase-mid 0.4 --phase-hard 0.2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

# Allow `python create_mixture_data_and_csv.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.corpus import (  # noqa: E402
    index_librispeech,
    index_noise,
    index_rirs,
    index_voxceleb2,
)


# ---------------------------------------------------------------------------
# Audio I/O (soundfile, no torchaudio dependency -- keeps the generator
# independent of the training loop)
# ---------------------------------------------------------------------------
def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    if sr != target_sr:
        # Cheap fallback: resample with librosa if available, else fail loudly.
        try:
            import librosa
            data = librosa.resample(data[0], orig_sr=sr, target_sr=target_sr)[None]
        except ImportError as e:
            raise RuntimeError(
                f"{path} has sr={sr}, target=16000. Install librosa or pre-resample."
            ) from e
    return data  # (1, T)


def save_wav(path: str, wav: np.ndarray, sr: int = 16000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav.squeeze(0), sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# Segment extraction + level scaling
# ---------------------------------------------------------------------------
def random_segment(wav: np.ndarray, length: int) -> np.ndarray:
    """Random crop of length `length`, zero-padded if shorter."""
    if wav.shape[1] >= length:
        start = random.randint(0, wav.shape[1] - length)
        return wav[:, start : start + length]
    pad = length - wav.shape[1]
    return np.pad(wav, ((0, 0), (0, pad)))


def scale_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_clean = np.mean(clean ** 2) + 1e-8
    p_noise_target = p_clean / (10 ** (snr_db / 10))
    p_noise = np.mean(noise ** 2) + 1e-8
    return noise * np.sqrt(p_noise_target / p_noise)


def scale_interferer(target: np.ndarray, interferer: np.ndarray, sir_db: float) -> np.ndarray:
    p_target = np.mean(target ** 2) + 1e-8
    p_int_target = p_target / (10 ** (sir_db / 10))
    p_int = np.mean(interferer ** 2) + 1e-8
    return interferer * np.sqrt(p_int_target / p_int)


def convolve_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Apply a room impulse response; trim/pad to clean's length."""
    reverbed = np.zeros_like(clean)
    for c in range(clean.shape[0]):
        x = np.convolve(clean[c], rir, mode="full")[: clean.shape[1]]
        reverbed[c] = x
    return reverbed


# ---------------------------------------------------------------------------
# Enrollment assembly (mirrors the original behaviour but supports VoxCeleb)
# ---------------------------------------------------------------------------
def assemble_enrollment(
    enrollment_pool: dict[str, list[str]],
    exclude_speaker: str,
    rng: random.Random,
    fixed_seconds: float = 7.0,
) -> np.ndarray:
    """Pick one utterance from a random *other* speaker if possible,
    otherwise from the target speaker. Truncates / zero-pads to length."""
    candidates = []
    if exclude_speaker in enrollment_pool and enrollment_pool[exclude_speaker]:
        # Prefer a different speaker for the enrollment when possible.
        for spk in enrollment_pool:
            if spk != exclude_speaker and enrollment_pool[spk]:
                candidates.extend(enrollment_pool[spk])
    if not candidates:
        candidates = enrollment_pool.get(exclude_speaker, [])

    if not candidates:
        # Last resort: zero enrollment.
        return np.zeros((1, int(fixed_seconds * 16000)), dtype=np.float32)

    path = rng.choice(candidates)
    wav = load_wav(path)
    target_len = int(fixed_seconds * 16000)
    if wav.shape[1] >= target_len:
        start = rng.randint(0, wav.shape[1] - target_len)
        return wav[:, start : start + target_len]
    return np.pad(wav, ((0, 0), (0, target_len - wav.shape[1])))


# ---------------------------------------------------------------------------
# Mixture generation
# ---------------------------------------------------------------------------
@dataclass
class PhaseSpec:
    name: str
    sir_range: tuple[float, float]
    snr_range: tuple[float, float]
    reverb_prob: float = 0.0
    num_mixtures: int = 0


def generate_phase(
    phase: PhaseSpec,
    corpora: dict,
    output_dir: str,
    segment_length: int,
    rng: random.Random,
) -> str:
    """Build `phase.num_mixtures` mixtures and write a CSV; returns csv path."""
    target_pool = corpora["target"]
    interferer_pool = corpora["interferer"]
    enrollment_pool = corpora["enrollment"]
    noise_pool = corpora.get("noise", [])
    rir_pool = corpora.get("rir", [])

    speaker_ids = list(target_pool.keys())
    if len(speaker_ids) < 2:
        raise ValueError("Need >= 2 speakers in the target pool.")

    mix_dir = os.path.join(output_dir, phase.name, "mixtures")
    enr_dir = os.path.join(output_dir, phase.name, "enrollment")
    tgt_dir = os.path.join(output_dir, phase.name, "target")
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(enr_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    rows = []
    for i in range(phase.num_mixtures):
        # Pick two distinct speakers
        target_spk, interferer_spk = rng.sample(speaker_ids, 2)

        target_files = target_pool[target_spk]
        interferer_files = interferer_pool[interferer_spk]
        if not target_files or not interferer_files:
            continue

        try:
            target_wav = random_segment(load_wav(rng.choice(target_files)), segment_length)
            interferer_wav = random_segment(
                load_wav(rng.choice(interferer_files)), segment_length
            )
        except Exception as e:
            print(f"  [skip] wav load error: {e}")
            continue

        sir_db = rng.uniform(*phase.sir_range)
        snr_db = rng.uniform(*phase.snr_range)
        interferer_scaled = scale_interferer(target_wav, interferer_wav, sir_db)

        if noise_pool and rng.random() < 0.7:  # 70% of mixtures have noise
            noise_wav = random_segment(load_wav(rng.choice(noise_pool)), segment_length)
            noise_scaled = scale_noise(target_wav, noise_wav, snr_db)
        else:
            noise_scaled = np.zeros_like(target_wav)

        # Optional reverberation on the target
        if rir_pool and rng.random() < phase.reverb_prob:
            rir = load_wav(rng.choice(rir_pool)).squeeze(0)
            target_wav = convolve_rir(target_wav, rir)

        mixture = target_wav + interferer_scaled + noise_scaled

        # Normalize to avoid clipping
        peak = float(np.max(np.abs(mixture)) + 1e-8)
        if peak > 0.99:
            mixture = mixture * (0.99 / peak)
            target_wav = target_wav * (0.99 / peak)

        enrollment_wav = assemble_enrollment(enrollment_pool, target_spk, rng)

        mix_path = os.path.join(mix_dir, f"mixture_{i:06d}.wav")
        enr_path = os.path.join(enr_dir, f"enrollment_{i:06d}.wav")
        tgt_path = os.path.join(tgt_dir, f"target_{i:06d}.wav")
        save_wav(mix_path, mixture)
        save_wav(enr_path, enrollment_wav)
        save_wav(tgt_path, target_wav)

        # Always use forward slashes (portable across macOS/Linux/Windows)
        rows.append({
            "mixture_path": mix_path.replace(os.sep, "/"),
            "enrollment_path": enr_path.replace(os.sep, "/"),
            "target_path": tgt_path.replace(os.sep, "/"),
            "sir_db": float(sir_db),
            "snr_db": float(snr_db),
        })

        if (i + 1) % 500 == 0:
            print(f"  [{phase.name}] {i + 1}/{phase.num_mixtures}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, phase.name, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [{phase.name}] wrote {len(df)} rows -> {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Config loading + corpus wiring
# ---------------------------------------------------------------------------
def load_corpora(cfg: dict, root_data: str) -> dict:
    """Index every corpus the config mentions; raise if a required one is missing."""
    target_name = cfg["data"]["target"]["corpus"]
    interferer_name = cfg["data"]["interferer"]["corpus"]
    enr_cfg = cfg["data"]["enrollment"]

    target_speakers = _index_speakers(target_name, root_data)
    interferer_speakers = (
        target_speakers if interferer_name == target_name
        else _index_speakers(interferer_name, root_data)
    )

    if enr_cfg["corpus"].startswith("voxceleb2"):
        # Mix VoxCeleb2 with the target corpus at the configured ratio.
        vox_speakers = _index_speakers("voxceleb2", root_data)
        if enr_cfg.get("voxceleb_fraction", 0.8) >= 1.0:
            enrollment_pool = vox_speakers
        else:
            # Keep a flat merged dict; the assembly function picks either.
            enrollment_pool = {**vox_speakers, **target_speakers}
    else:
        enrollment_pool = target_speakers

    corpora = {
        "target": target_speakers,
        "interferer": interferer_speakers,
        "enrollment": enrollment_pool,
    }
    if cfg["data"].get("noise"):
        corpora["noise"] = _index_noise(cfg["data"]["noise"]["corpus"], root_data)
    if cfg["data"].get("rir"):
        corpora["rir"] = _index_rirs(cfg["data"]["rir"]["corpus"], root_data)
    return corpora


def _index_speakers(name: str, root_data: str) -> dict[str, list[str]]:
    if name == "librispeech-clean-100":
        return index_librispeech(
            os.path.join(root_data, "LibriSpeech", "train-clean-100"),
            "librispeech-clean-100",
        )
    if name == "librispeech-clean-360":
        return index_librispeech(
            os.path.join(root_data, "LibriSpeech", "train-clean-360"),
            "librispeech-clean-360",
        )
    if name == "voxceleb2":
        return index_voxceleb2(os.path.join(root_data, "VoxCeleb2", "dev"))
    raise ValueError(f"Unknown speaker corpus: {name}")


def _index_noise(name: str, root_data: str) -> list[str]:
    if name == "wham":
        return index_noise(os.path.join(root_data, "wham_noise"), "wham_noise")
    if name == "dns4":
        # Legacy path; the original repo had `data/noise_fullband/`.
        return index_noise(
            os.path.join(root_data, "..", "noise_fullband"), "dns4_noise"
        )
    raise ValueError(f"Unknown noise corpus: {name}")


def _index_rirs(name: str, root_data: str) -> list[str]:
    if name == "whamr":
        return index_rirs(os.path.join(root_data, "wham_rir"), "wham_rir")
    raise ValueError(f"Unknown RIR corpus: {name}")


def load_config(path: str | None) -> dict:
    if path and os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f)
    # Built-in default
    return {
        "data": {
            "target":     {"corpus": "librispeech-clean-360"},
            "interferer": {"corpus": "librispeech-clean-360"},
            "enrollment": {"corpus": "voxceleb2-or-librispeech",
                           "voxceleb_fraction": 0.8},
            "noise":      {"corpus": "wham"},
            "rir":        {"corpus": "whamr", "prob": 0.0},
        },
        "mixture": {"segment_seconds": 4},
        "curriculum": {
            "phases": [
                {"name": "easy",  "sir_db": [0,  10],  "snr_db": [10, 25],
                 "reverb_prob": 0.0, "fraction": 0.20},
                {"name": "mid",   "sir_db": [-5, 5],   "snr_db": [5,  20],
                 "reverb_prob": 0.0, "fraction": 0.30},
                {"name": "hard",  "sir_db": [-10, 0],  "snr_db": [0,  15],
                 "reverb_prob": 0.0, "fraction": 0.30},
                {"name": "reverb","sir_db": [-10, 5],   "snr_db": [0,  20],
                 "reverb_prob": 0.3, "fraction": 0.20},
            ],
        },
    }


def phases_from_config(cfg: dict, num_mixtures: int) -> list[PhaseSpec]:
    raw = cfg["curriculum"]["phases"]
    if num_mixtures:
        # If the user passed an explicit total, distribute by `fraction`.
        for p in raw:
            p["num_mixtures"] = int(round(num_mixtures * p.get("fraction", 0)))
        # Fix rounding drift
        drift = num_mixtures - sum(p["num_mixtures"] for p in raw)
        raw[0]["num_mixtures"] += drift
    return [
        PhaseSpec(
            name=p["name"],
            sir_range=tuple(p["sir_db"]),
            snr_range=tuple(p["snr_db"]),
            reverb_prob=float(p.get("reverb_prob", 0.0)),
            num_mixtures=int(p.get("num_mixtures", num_mixtures // len(raw))),
        )
        for p in raw
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML curriculum config.")
    p.add_argument("--num-mixtures", type=int, default=0,
                   help="Total mixtures across all phases (0 = use per-phase num_mixtures).")
    p.add_argument("--output-dir", default="data_csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print("[load] indexing corpora ...")
    corpora = load_corpora(cfg, root_data)
    print(
        f"[load] speakers: target={len(corpora['target'])} "
        f"interferer={len(corpora['interferer'])} "
        f"enrollment={len(corpora['enrollment'])} "
        f"noise_files={len(corpora.get('noise', []))} "
        f"rirs={len(corpora.get('rir', []))}"
    )

    phases = phases_from_config(cfg, args.num_mixtures)
    rng = random.Random(args.seed)

    seg_seconds = cfg["mixture"].get("segment_seconds", 4)
    segment_length = int(seg_seconds * 16000)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[gen ] segment_length = {segment_length} samples ({seg_seconds}s)")
    for phase in phases:
        print(
            f"[gen ] phase={phase.name}  SIR={phase.sir_range}  "
            f"SNR={phase.snr_range}  reverb_prob={phase.reverb_prob}  "
            f"n={phase.num_mixtures}"
        )
        generate_phase(phase, corpora, args.output_dir, segment_length, rng)

    # Combined index CSV (for convenience)
    all_csvs = []
    for phase in phases:
        all_csvs.append(os.path.join(args.output_dir, phase.name, "metadata.csv"))
    combined = pd.concat([pd.read_csv(p) for p in all_csvs if os.path.isfile(p)],
                         ignore_index=True)
    combined.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
    print(f"[gen ] combined CSV: {os.path.join(args.output_dir, 'metadata.csv')} "
          f"({len(combined)} rows)")


if __name__ == "__main__":
    main()
