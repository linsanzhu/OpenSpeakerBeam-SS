"""
Split the HF 50k train set into a 4-phase curriculum.

The HF metadata.csv doesn't carry SIR/SNR labels (those are baked into the
mixtures), so we approximate difficulty by shuffling and partitioning the 50 000
rows. Phases 1..4 (easy/mid/hard/reverb) each contain 25/30/30/15 percent of
the data, in that order — the model sees every mixture exactly once across
all phases.

Train/dev split: 90 / 10 from the full 50 000.

Usage:
    python build_50k_curriculum.py
        # reads data/train/metadata.csv, writes:
        #   data_csv/train/metadata.csv       (all 45 000 rows, shuffled)
        #   data_csv/dev/metadata.csv         ( 5 000 rows, held-out)
        #   data_csv/curriculum/easy.csv      (11 250 rows)
        #   data_csv/curriculum/mid.csv       (13 500 rows)
        #   data_csv/curriculum/hard.csv      (13 500 rows)
        #   data_csv/curriculum/reverb.csv    ( 6 750 rows)
"""
import os, sys
import pandas as pd

SRC = "data/train/metadata.csv"
OUT_DIR = "data_csv"

# Phase weights, mirror configs/curriculum_50k.yaml
PHASES = [
    ("easy",   0.25),
    ("mid",    0.30),
    ("hard",   0.30),
    ("reverb", 0.15),
]

def main():
    if not os.path.isfile(SRC):
        sys.exit(f"missing {SRC}; extract train zips first.")
    df = pd.read_csv(SRC)
    n = len(df)
    print(f"loaded {n} rows from {SRC}")

    # The HF metadata uses Windows-style paths into  data_csv\train\...
    # Our on-disk files are at  data/train/...  (set up by setup_50k.py).
    # Rewrite each column's basename so they actually resolve.
    def to_real(p: str, sub: str) -> str:
        base = os.path.basename(p.replace("\\", "/"))
        return f"data/train/{sub}/{base}"

    df["mixture_path"]    = [to_real(p, "mixtures")   for p in df["mixture_path"]]
    df["enrollment_path"] = [to_real(p, "enrollment") for p in df["enrollment_path"]]
    df["target_path"]     = [to_real(p, "target")     for p in df["target_path"]]

    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Train / dev split: 90 / 10
    cut = int(n * 0.90)
    train_df = df.iloc[:cut].reset_index(drop=True)
    dev_df   = df.iloc[cut:].reset_index(drop=True)
    os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "dev"),   exist_ok=True)
    train_df.to_csv(os.path.join(OUT_DIR, "train", "metadata.csv"), index=False)
    dev_df.to_csv  (os.path.join(OUT_DIR, "dev",   "metadata.csv"), index=False)
    print(f"train={len(train_df)} dev={len(dev_df)}")

    # Curriculum phases
    os.makedirs(os.path.join(OUT_DIR, "curriculum"), exist_ok=True)
    pos = 0
    for name, frac in PHASES:
        n_phase = int(round(len(train_df) * frac))
        # Adjust last bucket so we exactly cover all rows
        if name == PHASES[-1][0]:
            n_phase = len(train_df) - pos
        phase_df = train_df.iloc[pos:pos + n_phase].reset_index(drop=True)
        out = os.path.join(OUT_DIR, "curriculum", f"{name}.csv")
        phase_df.to_csv(out, index=False)
        print(f"  {name}: {len(phase_df):>6} rows -> {out}")
        pos += n_phase
    assert pos == len(train_df), (pos, len(train_df))

if __name__ == "__main__":
    main()