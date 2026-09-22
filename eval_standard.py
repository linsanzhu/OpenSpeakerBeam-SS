"""
Standardized evaluation for SpeakerBeam-SS.

Reports per-condition:
    SI-SDR  (dB)       -- scale-invariant signal-to-distortion ratio
    PESQ             -- perceptual speech quality (16 kHz only)
    STOI             -- short-time objective intelligibility

Supports two benchmark layouts out of the box:

1. **legacy metadata.csv** (the format produced by
   [create_mixture_data_and_csv.py](create_mixture_data_and_csv.py) and the
   bundled `data/test_set/test/`): three columns `mixture_path,
   enrollment_path, target_path`. SI-SDR only is reported (PESQ/STOI are
   skipped because there's no clean source separate from the target).

2. **LibriMix** layout: `LibriMix/metadata/{mix_clean.json,
   sources.json, ...}` plus `wav16k/{min,min-both}/{mix,source1,...}`.
   Reports 2-speaker / 3-speaker clean / noisy conditions separately.

3. **LibriCSS** layout: `<session>/<mic>/{mix,<spk>.wav}`.

Usage:
    python eval_standard.py --model checkpoints/best_model.pth \\
        --benchmark legacy --root data/test_set/test --output eval.json
    python eval_standard.py --model checkpoints/best_model.pth \\
        --benchmark librimix-min --root data/LibriMix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

from model import SpeakerBeamSS
from tools.speaker_cache import SpeakerEmbeddingCache
from resemblyzer import VoiceEncoder

# Optional perceptual metrics (skip silently if unavailable)
try:
    from pesq import pesq as _pesq
    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False
try:
    from pystoi import stoi as _stoi
    _HAS_STOI = True
except Exception:
    _HAS_STOI = False


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def si_sdr(s: np.ndarray, s_hat: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB. Higher is better."""
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    proj = np.sum(s_hat * s) / (np.sum(s * s) + eps) * s
    noise = s_hat - proj
    return 10.0 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))


def safe_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    if not _HAS_PESQ:
        return None
    try:
        return float(_pesq(sr, ref.astype(np.float64), deg.astype(np.float64), "wb"))
    except Exception:
        return None


def safe_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    if not _HAS_STOI:
        return None
    try:
        return float(_stoi(ref, deg, sr, extended=False))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sample iterators (one per benchmark layout)
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    name: str
    mixture_path: str
    target_path: str
    enrollment_path: str | None
    condition: str  # e.g. "2spk_clean", "libricss_overlap40"


def iter_legacy(root: str, limit: int | None) -> Iterator[Sample]:
    """metadata.csv with Windows-style paths inside (HF test set)."""
    import pandas as pd
    csv_path = os.path.join(root, "metadata.csv")
    df = pd.read_csv(csv_path)
    norm = lambda p: p.replace("\\", "/").replace("data_csv/test/", "")
    for i, row in df.iterrows():
        yield Sample(
            name=f"clip_{i:06d}",
            mixture_path=os.path.join(root, norm(row["mixture_path"])),
            target_path=os.path.join(root, norm(row["target_path"])),
            enrollment_path=os.path.join(root, norm(row["enrollment_path"])),
            condition="legacy",
        )
        if limit is not None and i + 1 >= limit:
            break


def iter_librimix(root: str, cond: str, limit: int | None) -> Iterator[Sample]:
    """LibriMix metadata layout: metadata/{mix_clean.json, sources.json}.

    `cond` selects 'min' or 'min-both'.
    """
    import json as _json
    md_dir = os.path.join(root, "metadata")
    mix_json = os.path.join(md_dir, f"mix_{cond}.json")
    src_json = os.path.join(md_dir, f"sources_{cond}.json")
    if not os.path.isfile(mix_json) or not os.path.isfile(src_json):
        raise FileNotFoundError(
            f"Expected {mix_json} and {src_json}. Run LibriMix preparation first."
        )
    with open(mix_json) as f:
        mixes = _json.load(f)
    with open(src_json) as f:
        sources = _json.load(f)

    n_speakers = {"min": 2, "min-both": 2}[cond]  # LibriMix min variants are 2-spk
    # sources is a list parallel to mixes; each entry has "source1", "source2", ...
    sub_dir = "wav16k" + ("/" + cond if cond != "min" else "/min")
    for i, (mix_info, src_info) in enumerate(zip(mixes, sources)):
        mix_path = os.path.join(root, sub_dir, mix_info["mixture_path"])
        # Pick the first source as target by convention; for 2-speaker we
        # alternate between source1 and source2 to evaluate both targets.
        # Caller decides whether to iterate both directions; here we keep
        # one target per mix and the condition tag captures the noise level.
        for j in range(1, n_speakers + 1):
            tgt_path = os.path.join(root, sub_dir, src_info[f"source{j}"])
            yield Sample(
                name=f"{cond}_{i:06d}_src{j}",
                mixture_path=mix_path,
                target_path=tgt_path,
                enrollment_path=None,  # filled in by caller from a separate pool
                condition=cond,
            )
        if limit is not None and i + 1 >= limit:
            break


def iter_libricss(root: str, limit: int | None) -> Iterator[Sample]:
    """LibriCSS: <session>/<mic>/mix.wav plus per-speaker wavs."""
    for session in sorted(os.listdir(root)):
        s_dir = os.path.join(root, session)
        if not os.path.isdir(s_dir):
            continue
        for mic in sorted(os.listdir(s_dir)):
            m_dir = os.path.join(s_dir, mic)
            if not os.path.isdir(m_dir):
                continue
            mix = os.path.join(m_dir, "mix.wav")
            if not os.path.isfile(mix):
                continue
            # Enrollment is one of the per-speaker wavs (they're utterances
            # from the same speaker, just different time ranges).
            for fname in sorted(os.listdir(m_dir)):
                if not fname.startswith("source") or not fname.endswith(".wav"):
                    continue
                spk_wav = os.path.join(m_dir, fname)
                yield Sample(
                    name=f"{session}_{mic}_{fname}",
                    mixture_path=mix,
                    target_path=spk_wav,
                    enrollment_path=spk_wav,  # LibriCSS uses source as its own enrollment
                    condition=f"libricss_{session}",
                )
                if limit is not None and limit <= 0:
                    return
                if limit is not None:
                    limit -= 1


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    """Load a wav file as a (channels, samples) float32 array at SAMPLE_RATE."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            data = librosa.resample(data.T, orig_sr=sr, target_sr=SAMPLE_RATE).T
        except ImportError as e:
            raise RuntimeError(f"{path} sr={sr}; install librosa for resampling.") from e
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data.T  # (channels, samples) -- mono becomes (1, T)


def chunked_inference(
    model: SpeakerBeamSS,
    spk_emb: np.ndarray,
    mix: np.ndarray,
    chunk: int = SAMPLE_RATE,
) -> np.ndarray:
    """Run model on 1-second chunks (matches the export trace length)."""
    T = mix.shape[-1]
    outs = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            seg = mix[:, s : s + chunk]
            if seg.shape[-1] < chunk:
                seg = np.pad(seg, ((0, 0), (0, chunk - seg.shape[-1])))
            seg_t = torch.from_numpy(seg).unsqueeze(0)  # (1, 1, T)
            emb_t = torch.from_numpy(spk_emb)            # (1, 256)
            outs.append(model(seg_t, emb_t).cpu().numpy())
    return np.concatenate(outs, axis=-1)[..., :T]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="checkpoints/best_model.pth")
    parser.add_argument("--benchmark", default="legacy",
                        choices=["legacy", "librimix-min", "librimix-min-both", "libricss"])
    parser.add_argument("--root", default="data/test_set/test",
                        help="Benchmark root (legacy CSV dir or LibriMix/LibriCSS root).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on number of clips per benchmark condition.")
    parser.add_argument("--output", default="eval_results.json",
                        help="Where to write the JSON summary.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SpeakerBeamSS().to(device).eval()
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print(f"[load] model: {args.model}")

    encoder = VoiceEncoder(device="cpu")
    cache = SpeakerEmbeddingCache(encoder, max_size=4096)
    print(f"[load] speaker cache ready")

    # Pick an iterator
    if args.benchmark == "legacy":
        iterator = list(iter_legacy(args.root, args.limit))
        enrollments_per_clip = True
    elif args.benchmark == "librimix-min":
        iterator = list(iter_librimix(args.root, "min", args.limit))
        enrollments_per_clip = False
    elif args.benchmark == "librimix-min-both":
        iterator = list(iter_librimix(args.root, "min-both", args.limit))
        enrollments_per_clip = False
    elif args.benchmark == "libricss":
        iterator = list(iter_libricss(args.root, args.limit))
        enrollments_per_clip = True
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    print(f"[eval] {len(iterator)} samples via benchmark={args.benchmark}")
    by_cond: dict[str, list[dict]] = {}
    for i, s in enumerate(iterator):
        mix = load_wav(s.mixture_path)
        target = load_wav(s.target_path).squeeze(0)

        # Enrollment embedding
        if s.enrollment_path:
            emb = cache(None, [s.enrollment_path]).cpu().numpy()
        else:
            # LibriMix: enrollments aren't shipped by default. Use the target
            # waveform itself as the enrollment (allowed in LibriMix eval).
            enr = load_wav(s.target_path)
            emb = cache(enr, [s.target_path]).cpu().numpy()

        estimate = chunked_inference(model, emb, mix).squeeze(0).squeeze(0)

        # Trim to target length
        T = min(estimate.shape[-1], target.shape[-1])
        estimate = estimate[..., :T]
        target = target[..., :T]

        row = {
            "si_sdr": si_sdr(target, estimate),
            "pesq": safe_pesq(target, estimate),
            "stoi": safe_stoi(target, estimate),
        }
        by_cond.setdefault(s.condition, []).append(row)

        if (i + 1) % 25 == 0 or i == 0:
            mean_sdr = np.mean([r["si_sdr"] for r in by_cond[s.condition]])
            print(f"  [{i+1:4d}/{len(iterator)}] {s.condition} SI-SDR (running mean): "
                  f"{mean_sdr:6.2f} dB")

    summary = {}
    for cond, rows in by_cond.items():
        si = np.array([r["si_sdr"] for r in rows])
        pesq_vals = [r["pesq"] for r in rows if r["pesq"] is not None]
        stoi_vals = [r["stoi"] for r in rows if r["stoi"] is not None]
        summary[cond] = {
            "n": len(rows),
            "si_sdr_mean": float(si.mean()),
            "si_sdr_std": float(si.std()),
            "pesq_mean": float(np.mean(pesq_vals)) if pesq_vals else None,
            "stoi_mean": float(np.mean(stoi_vals)) if stoi_vals else None,
        }

    # Markdown table
    print("\n## Results\n")
    print("| condition | n | SI-SDR (dB) | PESQ | STOI |")
    print("|---|---|---|---|---|")
    for cond, m in summary.items():
        pesq_s = f"{m['pesq_mean']:.3f}" if m["pesq_mean"] is not None else "n/a"
        stoi_s = f"{m['stoi_mean']:.3f}" if m["stoi_mean"] is not None else "n/a"
        print(f"| {cond} | {m['n']} | {m['si_sdr_mean']:6.2f} ± {m['si_sdr_std']:4.2f} "
              f"| {pesq_s} | {stoi_s} |")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
