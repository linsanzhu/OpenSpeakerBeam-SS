#!/usr/bin/env python
"""Download just train/mixtures.zip with resume support."""
import os, sys, time
import requests

os.environ.pop('HF_XET', None)

URL = 'https://hf-mirror.com/datasets/helloidea/OpenSpeakerBeam-SS-dataset/resolve/main/train/mixtures.zip'
OUT = '/tmp/sb_train_dl/train_mixtures.zip'

# Sanity: if file already complete, skip
EXPECTED = 29755723940  # bytes
if os.path.exists(OUT):
    cur = os.path.getsize(OUT)
    if cur == EXPECTED:
        print(f'  already complete ({cur} bytes); skipping.', flush=True)
        sys.exit(0)
    print(f'  partial file: {cur/1e9:.2f} GB / {EXPECTED/1e9:.2f} GB', flush=True)
else:
    cur = 0
    print(f'  starting fresh.', flush=True)

t0 = time.time()
try:
    headers = {}
    if cur > 0:
        headers['Range'] = f'bytes={cur}-'
        print(f'  resuming from {cur/1e9:.2f} GB', flush=True)
    with requests.get(URL, stream=True, timeout=30, headers=headers) as r:
        if cur > 0 and r.status_code == 200:
            # Server ignores Range; restart
            print('  server ignored Range; restarting', flush=True)
            cur = 0
            r.close()
            r = requests.get(URL, stream=True, timeout=30)
            r.raise_for_status()
            mode = 'wb'
        else:
            r.raise_for_status()
            mode = 'ab' if cur > 0 else 'wb'
        total = int(r.headers.get('content-length', 0)) + cur
        with open(OUT, mode) as fh:
            downloaded = cur
            last = time.time()
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last > 10:
                    speed = (downloaded - cur) / (now - t0) / 1e6
                    pct = downloaded / total * 100 if total else 0
                    print(f'    {downloaded/1e9:.2f}/{total/1e9:.2f} GB'
                          f' ({pct:5.1f}%) {speed:.1f} MB/s', flush=True)
                    last = now
    print(f'  OK in {time.time()-t0:.1f}s -> {OUT}', flush=True)
except Exception as e:
    print(f'  FAIL: {type(e).__name__}: {e}', flush=True)
    sys.exit(1)