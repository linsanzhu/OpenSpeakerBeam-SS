import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import SpeakerBeamSS
from tools import get_speaker_embeddings_batch
from tools.speaker_cache import SpeakerEmbeddingCache
from resemblyzer import VoiceEncoder
from losses import si_snr_loss, MRSTFTLoss, combined_loss


# ========================================
# 1. Dataset の定義
# ========================================
def _load_wav_torch(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load a wav file as a (1, T) float32 torch.Tensor at 16 kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data.T, orig_sr=sr, target_sr=target_sr).T
        except ImportError as e:
            raise RuntimeError(
                f"{path} sr={sr}; install librosa or pre-resample."
            ) from e
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return torch.from_numpy(data.T)  # (1, T)


class SpeechDataset(Dataset):
    """
    CSVファイルに記載された音声パスから、mixture, enrollment, target のペアを返す Dataset
    CSV ファイルは、少なくとも以下のカラムを含むものとする:
        - mixture_path
        - enrollment_path
        - target_path
    """

    def __init__(self, csv_file, transform=None, fixed_length: int | None = None):
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.fixed_length = fixed_length  # samples; truncate / pad to this if set

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mixture = _load_wav_torch(row["mixture_path"])
        enrollment = _load_wav_torch(row["enrollment_path"])
        target = _load_wav_torch(row["target_path"])

        if self.fixed_length is not None:
            mixture = self._fix(mixture, self.fixed_length)
            enrollment = self._fix(enrollment, self.fixed_length)
            target = self._fix(target, self.fixed_length)

        return mixture, enrollment, target, row["mixture_path"], row["enrollment_path"], row["target_path"]

    @staticmethod
    def _fix(wav: torch.Tensor, length: int) -> torch.Tensor:
        T = wav.shape[-1]
        if T >= length:
            return wav[..., :length]
        return torch.nn.functional.pad(wav, (0, length - T))


def pad_collate(batch):
    """Right-pad variable-length waveforms to the longest in the batch.

    Returns (mixture, enrollment, target, mix_paths, enr_paths, tgt_paths).
    Each waveform tensor has shape (B, 1, T_max)."""
    mics, enrs, tgts, mix_paths, enr_paths, tgt_paths = zip(*batch)
    max_T = max(m.shape[-1] for m in mics)
    def _pad(w, T):
        if w.shape[-1] < T:
            return torch.nn.functional.pad(w, (0, T - w.shape[-1]))
        return w[..., :T]
    mics = torch.stack([_pad(m, max_T) for m in mics], dim=0)
    enrs = torch.stack([_pad(e, max_T) for e in enrs], dim=0)
    tgts = torch.stack([_pad(t, max_T) for t in tgts], dim=0)
    return mics, enrs, tgts, list(mix_paths), list(enr_paths), list(tgt_paths)


# ========================================
# 2. 検証 / テスト時用の評価関数
# ========================================
@torch.no_grad()
def evaluate(model, dataloader, speaker_cache, device, loss_fn=None):
    """DevやTestでSI-SNRを計算する共通関数"""
    model.eval()
    total_loss = 0.0
    for mixture, enrollment, target, _, enr_paths, _ in dataloader:
        mixture = mixture.to(device)
        enrollment = enrollment.to(device)
        target = target.to(device)

        # Cache the d-vectors by enrollment file path.
        speaker_embeddings = speaker_cache(enrollment, enr_paths)
        output = model(mixture, speaker_embeddings)

        # SI-SNR loss wants (B, T); take the shortest of (output, target).
        T_out = min(output.shape[-1], target.shape[-1])
        output = output[..., :T_out].squeeze(1)
        target = target[..., :T_out].squeeze(1)

        if loss_fn is None:
            loss = si_snr_loss(target, output)
        else:
            loss, _ = loss_fn(target, output)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()  # ここで学習モードに戻す
    return avg_loss


# ========================================
# 3. メイン学習関数
# ========================================
def train_and_validate(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] using {device}")

    # ---------------------------
    # (A) DataLoader の準備
    # ---------------------------
    train_dataset = SpeechDataset(
        csv_file=args.train_csv,
        fixed_length=int(args.segment_seconds * 16000) if args.segment_seconds > 0 else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
    )

    dev_dataset = SpeechDataset(csv_file=args.dev_csv)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )

    # Speaker embedding cache -- the encoder itself stays a module-level singleton.
    speaker_encoder = VoiceEncoder(device="cpu")
    speaker_cache = SpeakerEmbeddingCache(
        speaker_encoder,
        max_size=args.cache_size,
        precomputed_dir=getattr(args, "precomputed_dir", "") or None,
    )

    # ---------------------------
    # (B) モデルやオプティマイザの定義
    # ---------------------------
    model = SpeakerBeamSS().to(device)
    if getattr(args, "init_ckpt", None) and os.path.isfile(args.init_ckpt):
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"[init] warm-started from {args.init_ckpt}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.num_epochs * len(train_loader)),
            eta_min=args.lr * 0.01,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=args.reduce_patience,
        )

    # Loss: either pure SI-SNR or SI-SNR + alpha * MR-STFT.
    if args.loss == "combined":
        loss_fn = lambda s, sh: combined_loss(s, sh, alpha=args.mr_stft_alpha,
                                              mr_stft=MRSTFTLoss().to(device))
    else:
        loss_fn = None

    # ---------------------------
    # 学習開始
    # ---------------------------
    model.train()
    global_step = 0
    best_dev_loss = float("inf")
    patience_count = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        # ---------------------------
        # (C) Trainエポック
        # ---------------------------
        for batch_idx, (mixture, enrollment, target,
                        _, enr_paths, _) in enumerate(train_loader):
            mixture = mixture.to(device, non_blocking=True)
            enrollment = enrollment.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            speaker_embeddings = speaker_cache(enrollment, enr_paths)

            optimizer.zero_grad()
            output = model(mixture, speaker_embeddings)

            T_out = min(output.shape[-1], target.shape[-1])
            output = output[..., :T_out].squeeze(1)
            target = target[..., :T_out].squeeze(1)

            if loss_fn is None:
                loss = si_snr_loss(target, output)
            else:
                loss, _ = loss_fn(target, output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Cosine LR steps every batch; ReduceLROnPlateau steps every epoch.
            if args.scheduler == "cosine":
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                stats = speaker_cache.stats()
                print(
                    f"[Train] Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, emb_cache_hit={stats['hit_rate']:.2f}"
                )

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # ---------------------------
        # (D) Devエポック
        # ---------------------------
        dev_loss = evaluate(model, dev_loader, speaker_cache, device, loss_fn=loss_fn)
        print(f"[Dev]   Epoch {epoch+1}/{args.num_epochs}, Dev Loss: {dev_loss:.4f}")

        if args.scheduler != "cosine":
            scheduler.step(dev_loss)

        # ---------------------------
        # (E) ベストモデルの更新 & 早期停止
        # ---------------------------
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_count = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"=> Best model updated! Dev Loss = {dev_loss:.4f}")
        else:
            patience_count += 1
            if args.early_stop_patience > 0 and patience_count >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

        print(f"[Train] Epoch {epoch+1} finished! Average Train Loss: {avg_train_loss:.4f}\n")

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No best model found (no improvement on Dev set).")

    return model


# ========================================
# 4. テスト時の評価関数
# ========================================
def test_model(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] using {device}")

    test_dataset = SpeechDataset(csv_file=args.test_csv)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )

    speaker_encoder = VoiceEncoder(device="cpu")
    speaker_cache = SpeakerEmbeddingCache(
        speaker_encoder,
        max_size=args.cache_size,
        precomputed_dir=getattr(args, "precomputed_dir", "") or None,
    )
    model = SpeakerBeamSS().to(device)

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Train the model first.")

    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} for testing.")

    test_loss = evaluate(model, test_loader, speaker_cache, device, loss_fn=None)
    print(f"[Test] Test Loss (SI-SNR): {test_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train & Validate SpeakerBeam-SS")
    parser.add_argument("--train_csv", type=str, default="data_csv/train/metadata.csv")
    parser.add_argument("--dev_csv", type=str, default="data_csv/dev/metadata.csv")
    parser.add_argument("--test_csv", type=str, default="data_csv/test/metadata.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--segment_seconds", type=float, default=4.0,
                        help="Crop each clip to this length (0 = no cropping).")

    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["sisnr_only", "combined"],
                        help="Pure SI-SNR or SI-SNR + alpha * MR-STFT.")
    parser.add_argument("--mr_stft_alpha", type=float, default=0.3)

    # Scheduler / early stop
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="0 disables early stopping (curriculum runs the full epoch budget).")
    parser.add_argument("--reduce_patience", type=int, default=20)

    # Speaker embedding cache
    parser.add_argument("--cache_size", type=int, default=4096)
    parser.add_argument("--precomputed_dir", type=str, default="data/embeddings",
                        help="Directory with precomputed enrollment_embeddings.npy "
                             "and enrollment_paths.txt. Empty disables disk cache.")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="Optional checkpoint to warm-start from.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device (default: auto -> cuda > mps > cpu).")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_validate(args)
    elif args.mode == "test":
        test_model(args)
