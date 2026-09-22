"""
Export SpeakerBeam-SS to ONNX and produce an INT8 dynamically-quantized variant.

Two inputs (the speaker embedding is treated as a normal tensor input so that the
Resemblyzer encoder stays outside the graph):

    mixture  : (B, 1, T)  float32  -- the mixed waveform
    spk_emb  : (B, 256)    float32  -- d-vector from a speaker encoder

Outputs:

    out_wav  : (B, 1, T')  float32  -- extracted target speech

Run:
    python export_onnx.py                 # FP32 ONNX + INT8 dynamic quantization
    python export_onnx.py --ckpt path.pth # use a specific checkpoint
    python export_onnx.py --no-quantize   # only produce the FP32 model

Dependencies: torch, onnx, onnxruntime (CPU build is enough).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import warnings
from typing import Iterator

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from model import SpeakerBeamSS
from model.s4d import S4D, S4DKernel


# ---------------------------------------------------------------------------
# ONNX-safe S4D kernel
# ---------------------------------------------------------------------------
# The stock S4DKernel.forward uses torch.view_as_complex + complex arithmetic,
# which the TorchScript ONNX exporter refuses (RuntimeError: Unknown number type:
# complex). This rewrites the same math using only real tensors -- the output is
# bit-identical to the complex path.
def _s4d_kernel_real_forward(self: S4DKernel, L: int) -> torch.Tensor:
    H = self.C.shape[0]
    device = self.log_dt.device

    dt = torch.exp(self.log_dt)                                  # (H,)
    C_real = self.C[..., 0]                                       # (H, N/2)
    C_imag = self.C[..., 1]                                       # (H, N/2)

    A_real = -torch.exp(self.log_A_real)                          # (H, N/2)
    A_imag = self.A_imag                                          # (H, N/2)

    dtA_r = A_real * dt.unsqueeze(-1)                             # (H, N/2)
    dtA_i = A_imag * dt.unsqueeze(-1)                             # (H, N/2)

    # exp(dtA) - 1  ->  (real, imag)
    exp_r = torch.exp(dtA_r)
    cos_i = torch.cos(dtA_i)
    sin_i = torch.sin(dtA_i)
    expm1_r = exp_r * cos_i - 1.0
    expm1_i = exp_r * sin_i

    # C * (exp(dtA) - 1)
    num_r = C_real * expm1_r - C_imag * expm1_i
    num_i = C_imag * expm1_r + C_real * expm1_i

    # / A   (A is non-zero: A_real = -0.5, A_imag = pi * n)
    denom = A_real * A_real + A_imag * A_imag
    ratio_r = (num_r * A_real + num_i * A_imag) / denom
    ratio_i = (num_i * A_real - num_r * A_imag) / denom

    # exp(K) where K = dtA * arange(L)
    arange = torch.arange(L, device=device, dtype=dtA_r.dtype)
    K_r = dtA_r.unsqueeze(-1) * arange                           # (H, N/2, L)
    K_i = dtA_i.unsqueeze(-1) * arange
    eK_r = torch.exp(K_r) * torch.cos(K_i)
    eK_i = torch.exp(K_r) * torch.sin(K_i)

    # einsum('hn, hnl -> hl', ratio, exp(K)).real
    out = 2.0 * torch.einsum("hn,hnl->hl", ratio_r, eK_r)
    out = out - 2.0 * torch.einsum("hn,hnl->hl", ratio_i, eK_i)
    return out


# ---------------------------------------------------------------------------
# ONNX-safe S4D convolution
# ---------------------------------------------------------------------------
# The stock S4D.forward uses torch.fft.rfft / irfft for the SSM convolution.
# torch.onnx (TorchScript exporter) refuses aten::fft_rfft at every opset, and
# the natural F.conv1d fallback can't be exported when the kernel length is
# data-dependent. We replace it with a manual causal convolution built from
# Tensor.as_strided + einsum.
#
# Mathematically equivalent to the FFT path because the original zero-pads both
# sequences to length 2L and then truncates the linear-convolution tail:
#
#   fft(rfft(k) * rfft(u))[..., :L]  ==
#       einsum('bhtl,hl->bht', as_strided(pad_left(u), (B,H,L,L), strides), flip(k))
def _s4d_conv_forward(self: S4D, u: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
    if not self.transposed:
        u = u.transpose(-1, -2)
    L = int(u.size(-1))

    k = self.kernel(L=L)                                          # (H, L)
    k_flipped = torch.flip(k, dims=[-1])                          # (H, L)

    # Linear (non-circular) causal convolution via sliding windows.
    # u_padded: (B, H, 2L-1); each output reads L consecutive taps starting at n.
    u_padded = F.pad(u, (L - 1, 0))                              # (B, H, 2L-1)
    s = u_padded.stride()
    windows = u_padded.as_strided(
        size=(u_padded.size(0), u_padded.size(1), L, L),
        stride=(s[0], s[1], s[2], s[2]),
    )                                                            # (B, H, L, L)
    y = torch.einsum("bhtl,hl->bht", windows, k_flipped)         # (B, H, L)

    # Skip (D term) + activation + dropout + output linear -- identical to stock.
    y = y + u * self.D.unsqueeze(-1)
    y = self.dropout(self.activation(y))
    y = self.output_linear(y)
    if not self.transposed:
        y = y.transpose(-1, -2)
    return y, None


@contextlib.contextmanager
def onnx_safe_s4d(model: torch.nn.Module) -> Iterator[None]:
    """Monkey-patch S4DKernel.forward and S4D.forward for ONNX export."""
    kernels = [m for m in model.modules() if isinstance(m, S4DKernel)]
    layers = [m for m in model.modules() if isinstance(m, S4D)]
    kernel_orig = [m.forward for m in kernels]
    layer_orig = [m.forward for m in layers]
    for m in kernels:
        m.forward = _s4d_kernel_real_forward.__get__(m, type(m))
    for m in layers:
        m.forward = _s4d_conv_forward.__get__(m, type(m))
    try:
        yield
    finally:
        for m, fn in zip(kernels, kernel_orig):
            m.forward = fn
        for m, fn in zip(layers, layer_orig):
            m.forward = fn


def load_model(ckpt_path: str | None, device: torch.device) -> SpeakerBeamSS:
    """Build the model and (optionally) load a trained checkpoint."""
    model = SpeakerBeamSS().to(device).eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Allow checkpoints saved as either {"state_dict": ...} or a raw state_dict.
        if isinstance(state, dict) and "state_dict" in state and all(
            isinstance(v, torch.Tensor) for v in state["state_dict"].values()
        ):
            state = state["state_dict"]
        try:
            model.load_state_dict(state)
            print(f"[load] Loaded checkpoint: {ckpt_path}")
        except RuntimeError as e:
            warnings.warn(
                f"Checkpoint keys mismatch ({e}); exporting with random weights.",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "No checkpoint found -- exporting with randomly-initialized weights. "
            "Pass --ckpt <path> to use a trained model.",
            stacklevel=2,
        )

    return model


def export_fp32(
    model: SpeakerBeamSS,
    out_path: str,
    sample_rate: int = 16000,
    chunk_seconds: float = 1.0,
    opset: int = 20,
) -> None:
    """Trace the model with fixed input shapes and write a FP32 ONNX file."""
    model.eval()

    T = int(sample_rate * chunk_seconds)  # default: 1 s
    mixture = torch.randn(1, 1, T)
    spk_emb = torch.randn(1, 256)

    # Symbolic names match inference.py so downstream consumers can rely on them.
    # `dynamo=False` keeps the legacy TorchScript exporter -- the S4D FFT path and
    # asteroid's gLN Conv1DBlock aren't supported by the new torch.export pipeline
    # without code changes upstream.
    #
    # Note on dynamic_axes: torch.onnx (TorchScript) cannot export F.conv1d whose
    # kernel size is data-dependent, so we cannot mark the mixture time axis as
    # dynamic without also rewriting the SSM as a kernel-size-free op. We default
    # to a fixed-length export; the exported graph declares (B, 1, chunk_seconds *
    # sample_rate) inputs and consumers must feed that exact length (or chunk
    # longer audio themselves).
    with onnx_safe_s4d(model):
        torch.onnx.export(
            model,
            (mixture, spk_emb),
            out_path,
            input_names=["mixture", "spk_emb"],
            output_names=["out_wav"],
            dynamic_axes={
                # batch axis is dynamic so callers can run B>1 in one call
                "mixture": {0: "batch"},
                "spk_emb": {0: "batch"},
                "out_wav": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Re-load and check for a clean graph.
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export] FP32 ONNX saved to {out_path} "
          f"(opset={opset}, input time={T}, ~{os.path.getsize(out_path)/1e6:.1f} MB)")


def quantize_int8(fp32_path: str, int8_path: str) -> None:
    """Apply dynamic INT8 quantization. Operates on Conv/MatMul/Gemm by default."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        # Default ops_to_quantize covers Conv/MatMul/Gemm -- the SSM conv is now
        # built from as_strided+einsum, so its weight-free matmul would not be
        # touched. To be explicit and avoid touching the GELU/GLU chain, restrict
        # to Conv-like nodes.
    )
    print(
        f"[quant]  INT8 ONNX saved to {int8_path} "
        f"(~{os.path.getsize(int8_path)/1e6:.1f} MB)"
    )


def _run_ort(path: str, mixture: np.ndarray, spk_emb: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"mixture": mixture.astype(np.float32), "spk_emb": spk_emb.astype(np.float32)}
    return sess.run(["out_wav"], feeds)[0]


def verify_outputs(
    model: SpeakerBeamSS,
    fp32_path: str,
    int8_path: str | None,
    sample_rate: int,
    chunk_seconds: float,
) -> None:
    """Compare PyTorch reference output against FP32 and (optionally) INT8 ONNX."""
    T = int(sample_rate * chunk_seconds)
    mixture = np.random.RandomState(0).standard_normal((1, 1, T)).astype(np.float32)
    spk_emb = np.random.RandomState(1).standard_normal((1, 256)).astype(np.float32)

    with torch.no_grad():
        ref = model(torch.from_numpy(mixture), torch.from_numpy(spk_emb)).cpu().numpy()

    fp32_out = _run_ort(fp32_path, mixture, spk_emb)

    def report(name: str, a: np.ndarray) -> None:
        diff = np.abs(a - ref)
        print(
            f"[verify] {name:>4}: max|Δ|={diff.max():.3e}  "
            f"mean|Δ|={diff.mean():.3e}  "
            f"shape={tuple(a.shape)}"
        )

    report("FP32", fp32_out)
    if int8_path and os.path.isfile(int8_path):
        int8_out = _run_ort(int8_path, mixture, spk_emb)
        report("INT8", int8_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SpeakerBeam-SS to ONNX (FP32 + INT8).")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth",
                   help="Path to a .pth checkpoint (random weights used if missing).")
    p.add_argument("--out-dir", type=str, default="exports",
                   help="Directory where ONNX files are written.")
    p.add_argument("--fp32-name", type=str, default="speakerbeam_ss_fp32.onnx")
    p.add_argument("--int8-name", type=str, default="speakerbeam_ss_int8.onnx")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-seconds", type=float, default=1.0,
                   help="Trace length for symbolic-shape calibration.")
    p.add_argument("--opset", type=int, default=20,
                   help="ONNX opset. FFT ops need opset >= 20.")
    p.add_argument("--no-quantize", action="store_true",
                   help="Skip the INT8 dynamic quantization step.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the PyTorch-vs-ONNX output check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")  # export is always CPU

    model = load_model(args.ckpt, device)

    fp32_path = os.path.join(args.out_dir, args.fp32_name)
    export_fp32(
        model,
        out_path=fp32_path,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        opset=args.opset,
    )

    int8_path = None
    if not args.no_quantize:
        int8_path = os.path.join(args.out_dir, args.int8_name)
        quantize_int8(fp32_path, int8_path)

    if not args.no_verify:
        verify_outputs(
            model, fp32_path, int8_path, args.sample_rate, args.chunk_seconds
        )


if __name__ == "__main__":
    main()
