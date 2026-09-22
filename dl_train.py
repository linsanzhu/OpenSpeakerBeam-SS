#!/usr/bin/env python
"""Download the HF 50k train set in chunks via the regular HTTP path."""
import os, sys, time

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ.pop('HF_XET', None)

from huggingface_hub import hf_hub_download

files = [
    'train/enrollment.zip',   # 16.7 GB (download first to validate)
    'train/target.zip',       # 15.9 GB
    'train/mixtures.zip',     # 29.7 GB (largest)
]

for f in files:
    for attempt in range(3):
        t0 = time.time()
        try:
            p = hf_hub_download(
                'helloidea/OpenSpeakerBeam-SS-dataset', f,
                repo_type='dataset',
                local_dir='/tmp/sb_train_dl',
            )
            print(f'  OK {f}: {time.time()-t0:.1f}s -> {p}', flush=True)
            break
        except Exception as e:
            print(f'  ATTEMPT {attempt+1} {f}: {type(e).__name__}: {str(e)[:200]}', flush=True)
            time.sleep(10)
    else:
        print(f'  GIVE UP {f}', flush=True)
        sys.exit(1)

print('done', flush=True)