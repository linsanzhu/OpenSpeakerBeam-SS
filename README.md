# OpenSpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling

This is an **independent implementation** of [SpeakerBeam-SS](https://arxiv.org/abs/2407.01857), a real-time target speaker extraction model combining lightweight Conv-TasNet and State Space Modeling (S4D). The goal is to achieve efficient and high-performance speaker separation on resource-constrained devices.

🚨 **Disclaimer:** This repository is **not affiliated** with the authors of the original paper. It is an independent reimplementation and may have differences from the paper's methodology. If you have suggestions for improvements, feel free to share them! 🚨

## ✅ Project Status

The **network model implementation, training, and test dataset preparation are complete**. A full training cycle has been conducted using datasets published on Hugging Face, and test results are available. Some architectural differences from the original paper may exist. Feedback and pull requests are welcome.

## 📖 Reference

- **Paper:** [SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling](https://arxiv.org/abs/2407.01857)

## 📌 Features

- Conv-TasNet-based architecture with **S4D blocks** for efficient temporal modeling
- **Multiplicative adaptation** with d-vector speaker embeddings
- **1D convolutional blocks** for feature extraction
- **ONNX Runtime support** for CPU acceleration (AVX2 / AVX-512)
- **Designed for real-time inference** on mobile and server environments

## 🔧 Installation

### Dependencies

Install required dependencies with:

```sh
pip install -r requirements.txt
```

## 🚀 Usage

### 🔊 Inference

Run speaker extraction on a given mixture and enrollment audio:

```sh
python inference.py \
  --mixture data/sample/mixture_000001.wav \
  --enrollment data/sample/enrollment_000001.wav \
  --output data/sample/result_000001.wav
```

### 📦 ONNX Export & INT8 Quantization

Produce a deployable ONNX graph (FP32 + dynamic INT8) under `exports/`:

```sh
python export_onnx.py --ckpt checkpoints/best_model.pth
# -> exports/speakerbeam_ss_fp32.onnx  (~33 MB)
# -> exports/speakerbeam_ss_int8.onnx  (~13 MB)
```

The script patches the S4D FFT path (which the TorchScript exporter refuses)
with an equivalent real-valued `as_strided + einsum` formulation, then runs
`onnxruntime.quantization.quantize_dynamic` on the FP32 graph. Dynamic axes
cover `batch` only; the time axis is fixed at `chunk_seconds * sample_rate`
(default 1 s) because the SSM kernel length is data-dependent.

### 🏋️ Training

```sh
python train.py --mode=train
```

The new training script ([train.py](train.py)) uses:

- **`losses.combined_loss`** = SI-SNR + α · MR-STFT (perceptual).
  Switch to pure SI-SNR via `--loss sisnr_only`; tune α with `--mr_stft_alpha`.
- **Cosine LR schedule** by default (`--scheduler cosine`); falls back to
  ReduceLROnPlateau if requested.
- **`tools/speaker_cache.SpeakerEmbeddingCache`** — LRU-cached d-vectors
  keyed by enrollment file path, so the same enrollment across epochs
  reuses its embedding (watch the `emb_cache_hit` log).
- **Padding collate** — variable-length clips are right-padded; enable
  fixed crops with `--segment_seconds 4`.

### 📈 Scaled training pipeline (multi-dataset, curriculum)

For users who want to go beyond the 50 k-mixture baseline:

```sh
# 1. Download datasets (LibriSpeech, WHAM! noise, optionally LibriMix)
python tools/download_datasets.py --datasets librispeech-clean-360 wham \
    librimix --yes

# 2. Generate curriculum-style mixtures (200 k across 4 phases)
python create_mixture_data_and_csv.py --num-mixtures 200000 \
    --output-dir data_csv --seed 42

# 3. Train with the combined loss + cosine LR
python train.py --mode=train --train_csv data_csv/metadata.csv \
    --dev_csv data_csv/dev/metadata.csv \
    --batch_size 32 --num_epochs 150 --loss combined \
    --scheduler cosine --segment_seconds 4

# 4. Standardized evaluation against LibriMix or the bundled test set
python eval_standard.py --model checkpoints/best_model.pth \
    --benchmark legacy --root data/test_set/test \
    --output eval_results.json
```

YAML curriculum config (`configs/curriculum.yaml` if present, else the
built-in default) controls the four phases:

| Phase   | SIR (dB) | SNR (dB) | Reverb prob | Share |
|---------|----------|----------|-------------|-------|
| easy    | 0 ~ +10  | 10 ~ 25  | 0 %         | 20 %  |
| mid     | −5 ~ +5  | 5 ~ 20   | 0 %         | 30 %  |
| hard    | −10 ~ 0  | 0 ~ 15   | 0 %         | 30 %  |
| reverb  | −10 ~ +5 | 0 ~ 20   | 30 %        | 20 %  |

Enrollment draws 80 % from VoxCeleb2 (same speaker ID, different
recording) and 20 % from LibriSpeech to break the leak from
"target's own file = enrollment".

### 🧪 Testing

```sh
python train.py --mode=test
```

For standardized metrics (SI-SDR + PESQ + STOI), use
[eval_standard.py](eval_standard.py) instead:

```sh
python eval_standard.py --model checkpoints/best_model.pth \
    --benchmark legacy --root data/test_set/test --limit 200 \
    --output eval_results.json
```

Training and testing CSV metadata files are automatically downloaded and stored from Hugging Face:

```text
--train_csv data_csv/train/metadata.csv
--dev_csv   data_csv/dev/metadata.csv
--test_csv  data_csv/test/metadata.csv
```

## 💾 Dataset & Checkpoints

- ✅ **Test dataset and pretrained model available on Hugging Face:**  
  https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset/tree/main

- ✅ **Pretrained model:** `checkpoints/best_model.pth`

- 🔍 **[Test] Test Loss (SI-SNR): -5.8925**  
  *(Note: current performance is modest; improvements are planned.)*

- Evaluation result samples:

[enrollment audio 1](data/sample/enrollment_000001.wav)
[mixture audio 1](data/sample/mixture_000001.wav)
[result audio 1](data/sample/result_000001.wav)

[enrollment audio 2](data/sample/enrollment_000002.wav)
[mixture audio 2](data/sample/mixture_000002.wav)
[result audio 2](data/sample/result_000002.wav)

## 💡 Performance

Initial FLOP measurements on 1-second input (16kHz):

```
FLOPs: 21.60G, Params: 7.64M
```

- Expected to run **in real-time on modern CPUs** with **AVX2 or AVX-512** optimizations.
- **Neon acceleration** planned for **iOS devices** via ONNX Runtime.

## 📌 TODO

- Validate output quality
- Optimize model for mobile deployment

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
You are free to use it in commercial and closed-source projects, including mobile apps.

## 🙌 Acknowledgments

This work is inspired by the original SpeakerBeam-SS paper and the Conv-TasNet framework.

🔹 **Speaker embeddings are generated using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer/).**

