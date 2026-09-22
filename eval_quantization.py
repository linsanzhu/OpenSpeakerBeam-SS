"""
Compare PyTorch vs ONNX FP32 vs ONNX INT8 on real test audio.

For each test clip, runs all three backends on the same mixture + enrollment
embedding and reports:
  - SI-SNR(dB) vs the clean target (separation quality)
  - per-backend max|Δ| vs PyTorch (numerical drift)

Usage:
    python eval_quantization.py --n 50
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from tools import get_speaker_embeddings_batch

from model import SpeakerBeamSS


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """soundfile-based loader to avoid torchaudio's torchcodec dependency."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.T, sr  # (channels, samples)


SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE  # 1-second chunks, matching the export trace length
MODEL_INPUT_NAME = "mixture"
SPK_INPUT_NAME = "spk_emb"
OUT_NAME = "out_wav"


def si_snr(s: np.ndarray, s_hat: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB. Inputs are (T,) float arrays."""
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    proj = np.sum(s_hat * s) / (np.sum(s * s) + eps) * s
    noise = s_hat - proj
    return 10.0 * np.log10(
        (np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)
    )


def chunked_inference_torch(model: SpeakerBeamSS, mix: torch.Tensor, emb: torch.Tensor) -> np.ndarray:
    """Process the whole waveform in 1-s chunks (same chunk length as export)."""
    B, _, T = mix.shape
    outs = []
    with torch.no_grad():
        for start in range(0, T, CHUNK):
            seg = mix[:, :, start : start + CHUNK]
            if seg.shape[-1] < CHUNK:
                seg = torch.nn.functional.pad(seg, (0, CHUNK - seg.shape[-1]))
            outs.append(model(seg, emb).cpu().numpy())
    return np.concatenate(outs, axis=-1)[:, :, :T]


def chunked_inference_ort(sess: ort.InferenceSession, mix: np.ndarray, emb: np.ndarray) -> np.ndarray:
    B, _, T = mix.shape
    outs = []
    for start in range(0, T, CHUNK):
        seg = mix[:, :, start : start + CHUNK].astype(np.float32)
        if seg.shape[-1] < CHUNK:
            seg = np.pad(seg, ((0, 0), (0, 0), (0, CHUNK - seg.shape[-1])))
        outs.append(
            sess.run([OUT_NAME], {MODEL_INPUT_NAME: seg, SPK_INPUT_NAME: emb.astype(np.float32)})[0]
        )
    return np.concatenate(outs, axis=-1)[:, :, :T]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_model.pth")
    p.add_argument("--fp32-onnx", default="exports/speakerbeam_ss_fp32.onnx")
    p.add_argument("--int8-onnx", default="exports/speakerbeam_ss_int8.onnx")
    p.add_argument("--test-dir", default="data/test_set/test")
    p.add_argument("--n", type=int, default=50, help="Number of test clips to evaluate")
    p.add_argument("--limit-seconds", type=float, default=4.0,
                   help="Trim each clip to this length to keep the eval fast.")
    args = p.parse_args()

    device = torch.device("cpu")
    speaker_encoder = VoiceEncoder(device="cpu")

    # ---- Load backends ----
    print("[load] PyTorch model")
    torch_model = SpeakerBeamSS().to(device).eval()
    torch_model.load_state_dict(torch.load(args.ckpt, map_location=device))

    print(f"[load] ONNX FP32: {args.fp32_onnx}")
    sess_fp32 = ort.InferenceSession(args.fp32_onnx, providers=["CPUExecutionProvider"])

    print(f"[load] ONNX INT8: {args.int8_onnx}")
    sess_int8 = ort.InferenceSession(args.int8_onnx, providers=["CPUExecutionProvider"])

    # ---- Iterate over test clips ----
    csv_path = os.path.join(args.test_dir, "metadata.csv")
    with open(csv_path) as f:
        lines = [ln.strip().split(",") for ln in f.readlines()[1:] if ln.strip()]
    # CSV uses Windows-style paths ("data_csv\\test\\...") -- normalize to POSIX.
    norm = lambda p: p.replace("\\", "/").replace("data_csv/test/", "")
    rows = [(norm(m), norm(e), norm(t)) for m, e, t in lines][: args.n]

    metrics: Dict[str, list] = {
        "si_torch": [], "si_fp32": [], "si_int8": [],
        "drift_fp32": [], "drift_int8": [],
    }
    max_samples = int(args.limit_seconds * SAMPLE_RATE)

    for i, (mix_path, enr_path, tgt_path) in enumerate(rows):
        mix, sr = load_wav(os.path.join(args.test_dir, mix_path))
        assert sr == SAMPLE_RATE, f"unexpected sr={sr}"
        if mix.shape[0] > 1:
            mix = mix.mean(0, keepdim=True)
        mix_t = torch.from_numpy(mix).unsqueeze(0)  # (1, 1, T)
        if mix_t.shape[-1] > max_samples:
            mix_t = mix_t[..., :max_samples]

        target, _ = load_wav(os.path.join(args.test_dir, tgt_path))
        if target.shape[0] > 1:
            target = target.mean(0)
        else:
            target = target[0]
        target = target[: mix_t.shape[-1]]

        enr, _ = load_wav(os.path.join(args.test_dir, enr_path))
        if enr.shape[0] > 1:
            enr = enr.mean(0, keepdim=True)
        enr_t = torch.from_numpy(enr).unsqueeze(0)  # (1, 1, T_enr)

        emb = get_speaker_embeddings_batch(speaker_encoder, enr_t).numpy()

        # Run all three backends on identical inputs.
        out_torch = chunked_inference_torch(torch_model, mix_t, torch.from_numpy(emb))
        mix_np = mix_t.numpy()
        out_fp32 = chunked_inference_ort(sess_fp32, mix_np, emb)
        out_int8 = chunked_inference_ort(sess_int8, mix_np, emb)

        # SI-SNR vs clean target
        t_torch = out_torch.squeeze()
        t_fp32 = out_fp32.squeeze()
        t_int8 = out_int8.squeeze()
        metrics["si_torch"].append(si_snr(target, t_torch))
        metrics["si_fp32"].append(si_snr(target, t_fp32))
        metrics["si_int8"].append(si_snr(target, t_int8))

        # Numerical drift vs PyTorch reference
        metrics["drift_fp32"].append(float(np.abs(out_fp32 - out_torch).max()))
        metrics["drift_int8"].append(float(np.abs(out_int8 - out_torch).max()))

        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"[{i+1:3d}/{len(rows)}] "
                f"SI-SNR torch={metrics['si_torch'][-1]:6.2f}  "
                f"fp32={metrics['si_fp32'][-1]:6.2f}  "
                f"int8={metrics['si_int8'][-1]:6.2f} dB | "
                f"drift fp32={metrics['drift_fp32'][-1]:.2e}  "
                f"int8={metrics['drift_int8'][-1]:.2e}"
            )

    print("\n=== Summary over {} clips ===".format(len(rows)))
    for k, v in metrics.items():
        v = np.array(v)
        print(f"  {k:11s}  mean={v.mean():7.3f}  median={np.median(v):7.3f}  max={v.max():7.3f}")


if __name__ == "__main__":
    main()
