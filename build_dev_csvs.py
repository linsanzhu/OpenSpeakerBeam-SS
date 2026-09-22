"""Build train/dev CSVs from data/dev/ (the HF dev.zip extraction)."""
import os, pandas as pd

SRC = "data/dev/metadata.csv"
OUT_DIR = "data_csv"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)
print(f"loaded {len(df)} rows from {SRC}")

# The HF metadata has paths like  data_csv\dev\mixtures\mixture_000000.wav
# Strip everything up to and including the last \\, then prepend the real
# on-disk location which is  data/dev/{mixtures,enrollment,target}/
def to_real(hf_path: str, sub: str) -> str:
    base = os.path.basename(hf_path.replace("\\", "/"))
    return f"data/dev/{sub}/{base}"

out_df = pd.DataFrame({
    "mixture_path":    [to_real(r, "mixtures")   for r in df["mixture_path"]],
    "enrollment_path": [to_real(r, "enrollment") for r in df["enrollment_path"]],
    "target_path":     [to_real(r, "target")     for r in df["target_path"]],
})

n = len(out_df)
cut = int(n * 0.8)
train_df = out_df.iloc[:cut].reset_index(drop=True)
dev_df   = out_df.iloc[cut:].reset_index(drop=True)

os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "dev"),   exist_ok=True)
train_csv = os.path.join(OUT_DIR, "train", "metadata.csv")
dev_csv   = os.path.join(OUT_DIR, "dev",   "metadata.csv")
train_df.to_csv(train_csv, index=False)
dev_df.to_csv(dev_csv, index=False)
print(f"train ({len(train_df)}) -> {train_csv}")
print(f"dev   ({len(dev_df)}) -> {dev_csv}")
print("\nsample rows:")
print(train_df.head(2).to_string())