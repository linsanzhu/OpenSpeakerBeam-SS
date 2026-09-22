"""
Move the HF 50 k train set from /tmp/sb_train_dl/ into data/train/ inside the
project. Idempotent: skips files that already exist.

After running this, the project layout is:

    data/train/metadata.csv
    data/train/mixtures/mixture_000000.wav ...
    data/train/enrollment/enrollment_000000.wav ...
    data/train/target/target_000000.wav ...
"""
import os, shutil, sys

SRC = '/tmp/sb_train_dl'
DST = 'data/train'
os.makedirs(os.path.join(DST, 'mixtures'),   exist_ok=True)
os.makedirs(os.path.join(DST, 'enrollment'), exist_ok=True)
os.makedirs(os.path.join(DST, 'target'),     exist_ok=True)

# Copy metadata.csv
md_src = os.path.join(SRC, 'train_metadata.csv')
md_dst = os.path.join(DST, 'metadata.csv')
shutil.copy(md_src, md_dst)
print(f'metadata -> {md_dst}')

# Walk extracted/ and copy files into data/train/
extracted = os.path.join(SRC, 'extracted')
if not os.path.isdir(extracted):
    print(f'NOTE: {extracted} not present yet; zips still extracting.')

copied = {'enrollment': 0, 'target': 0, 'mixtures': 0}
for split in ('enrollment', 'target', 'mixtures'):
    src_dir = os.path.join(extracted, split)
    dst_dir = os.path.join(DST, split)
    if not os.path.isdir(src_dir):
        continue
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.exists(d):
            continue
        # Use hardlink or copy; hardlink saves disk and is instant
        try:
            os.link(s, d)
            copied[split] += 1
        except OSError:
            shutil.copy2(s, d)
            copied[split] += 1
    print(f'  {split}: {len(os.listdir(dst_dir))} files now in {dst_dir}')

print(f'\nDone. Disk after:')
os.system('du -sh data/train/')