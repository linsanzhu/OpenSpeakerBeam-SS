"""
Pre-compute d-vector embeddings for all unique enrollment files using a
multiprocessing pool (resemblyzer is CPU-bound and Python can fork safely).

Saves to:
    data/embeddings/enrollment_embeddings.npy    (N, 256) float32 memmap
    data/embeddings/enrollment_paths.txt         one path per line, same order
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


def collect_paths(train_csv: str, dev_csv: str) -> list[str]:
    paths = []
    for csv in (train_csv, dev_csv):
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        paths.extend(df["enrollment_path"].tolist())
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


_GLOBAL_ENCODER = None
def _init_worker():
    global _GLOBAL_ENCODER
    # Re-init resemblyzer in each worker
    _GLOBAL_ENCODER = VoiceEncoder(device="cpu")


def embed_one(path: str) -> tuple[int, np.ndarray]:
    global _GLOBAL_ENCODER
    try:
        wav, _ = sf.read(path, dtype="float32")
        pre = preprocess_wav(wav)
        emb = np.asarray(_GLOBAL_ENCODER.embed_utterance(pre), dtype=np.float32)
        return (0, emb)
    except Exception:
        return (0, np.zeros(256, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data_csv/train/metadata.csv")
    p.add_argument("--dev_csv",   default="data_csv/dev/metadata.csv")
    p.add_argument("--out_dir",   default="data/embeddings")
    p.add_argument("--workers",   type=int, default=4)
    args = p.parse_args()

    paths = collect_paths(args.train_csv, args.dev_csv)
    print(f"[collect] {len(paths)} unique enrollment paths")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_emb = os.path.join(out_dir, "enrollment_embeddings.npy")
    out_paths = os.path.join(out_dir, "enrollment_paths.txt")

    if os.path.isfile(out_emb) and os.path.isfile(out_paths):
        with open(out_paths) as f:
            existing = [l.rstrip() for l in f if l.strip()]
        if len(existing) == len(paths):
            print(f"  cache already complete ({len(existing)} entries)")
            return

    with open(out_paths, "w") as f:
        for p in paths:
            f.write(p + "\n")

    emb_arr = np.lib.format.open_memmap(
        out_emb, mode="w+", dtype=np.float32, shape=(len(paths), 256),
    )

    t0 = time.time()
    last_print = t0
    completed = 0

    # chunked submit to limit memory
    chunk = 200
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = []
        for i, pth in enumerate(paths):
            futs.append((i, ex.submit(embed_one, pth)))
            if len(futs) >= chunk:
                for ii, f in futs:
                    try:
                        _, emb = f.result()
                    except Exception as e:
                        emb = np.zeros(256, dtype=np.float32)
                    emb_arr[ii] = emb
                completed += len(futs)
                futs = []
                now = time.time()
                rate = completed / (now - t0)
                eta = (len(paths) - completed) / rate
                print(f"  [{completed}/{len(paths)}]  rate={rate:.2f} emb/s  ETA={eta/60:.1f} min", flush=True)
        for ii, f in futs:
            try:
                _, emb = f.result()
            except Exception:
                emb = np.zeros(256, dtype=np.float32)
            emb_arr[ii] = emb
        completed += len(futs)

    emb_arr.flush()
    print(f"\n[done] {completed} embeddings -> {out_emb}")
    print(f"  total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()