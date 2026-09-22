#!/usr/bin/env python
"""Download the HF 50k train set via hf-mirror.com (direct huggingface.co is
blocked on this network)."""
import os, sys, time
import requests

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ.pop('HF_XET', None)

MIRROR = 'https://hf-mirror.com/datasets/helloidea/OpenSpeakerBeam-SS-dataset/resolve/main'

files = [
    'train/enrollment.zip',   # 16.7 GB
    'train/target.zip',       # 15.9 GB
    'train/mixtures.zip',     # 29.7 GB
]

OUT_DIR = '/tmp/sb_train_dl'
os.makedirs(OUT_DIR, exist_ok=True)

for f in files:
    out = os.path.join(OUT_DIR, f.replace('/', '_'))
    # Resume from partial
    pos = os.path.getsize(out) if os.path.exists(out) else 0
    url = f'{MIRROR}/{f}'
    t0 = time.time()
    try:
        headers = {}
        if pos > 0:
            headers['Range'] = f'bytes={pos}-'
            print(f'    [resume {f}] from {pos/1e9:.2f} GB', flush=True)
        with requests.get(url, stream=True, timeout=30, headers=headers) as r:
            if pos > 0 and r.status_code != 206:
                # Server doesn't support range; restart
                print(f'    [no-range support; restart {f}]', flush=True)
                pos = 0
                r.close()
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                mode = 'wb'
            else:
                r.raise_for_status()
                mode = 'ab' if pos > 0 else 'wb'
            total = int(r.headers.get('content-length', 0)) + pos
            with open(out, mode) as fh:
                downloaded = pos
                last = time.time()
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last > 5:
                        speed = (downloaded - pos) / (now - t0) / 1e6
                        pct = downloaded / total * 100 if total else 0
                        print(f'    [{f}] {downloaded/1e9:.2f}/{total/1e9:.2f} GB'
                              f' ({pct:5.1f}%) {speed:.1f} MB/s', flush=True)
                        last = now
        print(f'  OK {f}: {time.time()-t0:.1f}s -> {out}', flush=True)
    except Exception as e:
        print(f'  FAIL {f}: {type(e).__name__}: {e}', flush=True)
        sys.exit(1)

print('done', flush=True)