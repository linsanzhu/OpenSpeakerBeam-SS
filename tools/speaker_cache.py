"""
LRU cache for d-vector (Resemblyzer) embeddings.

The original [tools/__init__.py](tools/__init__.py) recomputes the d-vector
for every enrollment waveform on every batch. Since a typical training run
loops over the same enrollment files many times per epoch, caching by
`(file_path, mtime)` cuts Resemblyzer work by an order of magnitude with
negligible memory (256 floats * 4k entries ~= 4 MB).

A second, larger cache is supported: if a precomputed memmap
(`enrollment_embeddings.npy` + `enrollment_paths.txt`) is present, the cache
loads embeddings from disk on a miss instead of running the encoder. This
is the fast path for long runs over a fixed enrollment set.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder, preprocess_wav


class SpeakerEmbeddingCache:
    """Drop-in replacement for `get_speaker_embeddings_batch` with caching.

    A precomputed memmap (`enrollment_embeddings.npy` + `enrollment_paths.txt`)
    can be attached so the cache skips the encoder entirely on hits.
    """

    def __init__(
        self,
        encoder: VoiceEncoder,
        max_size: int = 4096,
        precomputed_dir: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.max_size = max_size
        self._cache: OrderedDict[tuple[str, float], np.ndarray] = OrderedDict()
        # Hit/miss counters for diagnostics.
        self.hits = 0
        self.misses = 0

        # Optional precomputed disk cache. On a memory-cache miss we look up
        # the path in this memmap and short-circuit the encoder.
        self._disk_emb: np.memmap | None = None
        self._disk_paths: list[str] = []
        self._disk_index: dict[str, int] = {}
        self.disk_hits = 0
        if precomputed_dir is not None:
            self._load_disk_cache(precomputed_dir)

    def _load_disk_cache(self, precomputed_dir: str) -> None:
        emb_path = os.path.join(precomputed_dir, "enrollment_embeddings.npy")
        paths_path = os.path.join(precomputed_dir, "enrollment_paths.txt")
        if not (os.path.isfile(emb_path) and os.path.isfile(paths_path)):
            return
        try:
            self._disk_emb = np.load(emb_path, mmap_mode="r")
            with open(paths_path) as f:
                self._disk_paths = [line.rstrip("\n") for line in f if line.strip()]
            self._disk_index = {p: i for i, p in enumerate(self._disk_paths)}
            # Validate shapes match
            if self._disk_emb.shape != (len(self._disk_paths), 256):
                raise ValueError(
                    f"disk cache shape {self._disk_emb.shape} != "
                    f"({len(self._disk_paths)}, 256)"
                )
            # Reject cache if most rows are zero (interrupted precompute).
            sample = self._disk_emb[: min(200, len(self._disk_paths))]
            if (sample.std(axis=1) < 1e-6).mean() > 0.5:
                raise ValueError("disk cache mostly zero; precompute interrupted")
            print(
                f"[speaker_cache] loaded disk cache with "
                f"{len(self._disk_paths)} entries from {precomputed_dir}"
            )
        except Exception as e:
            print(f"[speaker_cache] disk cache unavailable: {e}")
            self._disk_emb = None
            self._disk_paths = []
            self._disk_index = {}

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def _key(self, path: str) -> tuple[str, float]:
        try:
            return (os.path.abspath(path), os.path.getmtime(path))
        except OSError:
            # File missing: use a sentinel so we re-attempt later.
            return (os.path.abspath(path), -1.0)

    def _embed_one(self, path: str) -> np.ndarray:
        # resemblyzer.preprocess_wav is a module-level helper, NOT a method
        # on VoiceEncoder. Calling self.encoder.preprocess_wav(...) silently
        # raises AttributeError and the except in get_batch then zero-fills.
        wav, _ = sf.read(path, dtype="float32")
        pre = preprocess_wav(wav)
        emb = np.asarray(self.encoder.embed_utterance(pre), dtype=np.float32)
        return emb

    def get_batch(
        self, wavs: torch.Tensor | None, paths: Iterable[str] | None,
    ) -> torch.Tensor:
        """Match the signature of `get_speaker_embeddings_batch`.

        Provide `paths` for cache benefit. `wavs` is accepted (and ignored
        when `paths` is given) so call-sites can be a drop-in replacement.
        """
        if paths is None:
            raise ValueError(
                "SpeakerEmbeddingCache requires file paths to key the cache. "
                "Pass paths=... or use tools.get_speaker_embeddings_batch instead."
            )

        paths = list(paths)
        assert len(paths) == wavs.shape[0] if wavs is not None else True, (
            "paths length must match batch size"
        )

        out = np.zeros((len(paths), 256), dtype=np.float32)
        for i, p in enumerate(paths):
            k = self._key(p)
            if k in self._cache:
                self._cache.move_to_end(k)
                out[i] = self._cache[k]
                self.hits += 1
                continue

            # Memory miss: try the disk cache first (path-indexed memmap).
            disk_idx = self._disk_index.get(p)
            if disk_idx is not None and self._disk_emb is not None:
                out[i] = np.asarray(self._disk_emb[disk_idx], dtype=np.float32)
                # Promote into memory LRU so subsequent hits stay hot.
                self._cache[k] = out[i]
                if len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
                self.disk_hits += 1
                self.misses += 1
                continue

            try:
                out[i] = self._embed_one(p)
            except Exception:
                # On failure, fall back to a zero embedding so the batch
                # still trains (matches previous behaviour).
                out[i] = np.zeros(256, dtype=np.float32)
            self._cache[k] = out[i]
            self.misses += 1
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

        device = wavs.device if wavs is not None else torch.device("cpu")
        return torch.from_numpy(out).to(device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.get_batch(*args, **kwargs)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "cached_entries": len(self._cache),
            "disk_hits": self.disk_hits,
        }
