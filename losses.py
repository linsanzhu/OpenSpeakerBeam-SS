"""
Training losses for SpeakerBeam-SS.

- si_snr_loss:        scale-invariant SNR (dB), negative for minimization.
- MRSTFTLoss:         multi-resolution STFT magnitude + log-magnitude L1.
- combined_loss:      weighted sum (SI-SNR + alpha * MR-STFT) -- the default.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def si_snr_loss(s: torch.Tensor, s_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative SI-SNR in dB. Inputs are (B, T) waveforms.

    Returns the *mean* over the batch; lower is better.
    """
    s = s - torch.mean(s, dim=1, keepdim=True)
    s_hat = s_hat - torch.mean(s_hat, dim=1, keepdim=True)

    s_target = (
        torch.sum(s_hat * s, dim=1, keepdim=True)
        / (torch.sum(s * s, dim=1, keepdim=True) + eps)
        * s
    )
    e_noise = s_hat - s_target
    ratio = torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    si_snr = 10.0 * torch.log10(ratio + eps)
    return -torch.mean(si_snr)


class MRSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.

    Sums magnitude-L1 and log-magnitude-L1 at several FFT resolutions; known
    to correlate better with perceptual quality than time-domain losses
    alone. Inspired by BigVGAN, HiFi-GAN, and modern speech synthesis.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] | None = None,
        win_sizes: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if hop_sizes is None:
            hop_sizes = tuple(s // 4 for s in fft_sizes)
        if win_sizes is None:
            win_sizes = fft_sizes
        assert len(hop_sizes) == len(fft_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        # Hann windows registered as buffers so they move with the module.
        self.register_buffer("_dummy", torch.zeros(1))  # ensures .to(device) works
        self._windows: list[torch.Tensor] = []  # lazily created on first forward
        self._windows_device: torch.device | None = None

    def _ensure_windows(self, device: torch.device) -> list[torch.Tensor]:
        if self._windows_device != device:
            self._windows = [
                torch.hann_window(w, device=device) for w in self.win_sizes
            ]
            self._windows_device = device
        return self._windows

    def forward(self, s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        """Inputs (B, T) waveforms."""
        if s.dim() == 2:
            s, s_hat = s.unsqueeze(1), s_hat.unsqueeze(1)  # (B, 1, T) for stft

        # MPS has incomplete support for complex STFT. Compute STFT on CPU
        # to keep this loss device-agnostic. The transfer is small (B*T float32)
        # and the cost is amortised over the FFT work.
        orig_device = s.device
        if orig_device.type == "mps":
            s    = s.detach().to("cpu")
            s_hat = s_hat.detach().to("cpu")
        s1    = s.squeeze(1)
        s_hat1 = s_hat.squeeze(1)

        windows = self._ensure_windows(s.device)
        total = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, windows):
            S  = torch.stft(s1, fft, hop, window=win, return_complex=True,
                            center=True, pad_mode="reflect")
            Sh = torch.stft(s_hat1, fft, hop, window=win, return_complex=True,
                            center=True, pad_mode="reflect")
            mag_s = S.abs()
            mag_sh = Sh.abs()
            total = total + (mag_s - mag_sh).abs().mean()
            total = total + (
                torch.log(mag_s + 1e-7) - torch.log(mag_sh + 1e-7)
            ).abs().mean()
        return (total / len(self.fft_sizes)).to(orig_device)


def combined_loss(
    s: torch.Tensor,
    s_hat: torch.Tensor,
    alpha: float = 0.3,
    mr_stft: nn.Module | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (loss, components) where components is a dict for logging."""
    sisnr = si_snr_loss(s, s_hat, eps)
    if mr_stft is None:
        mr = torch.zeros((), device=s.device)
    else:
        mr = mr_stft(s, s_hat)
    total = sisnr + alpha * mr
    return total, {"sisnr": sisnr.item(), "mr_stft": float(mr.item())}
