"""
Bf16 activation cache on NVMe using torch memory-mapped storage.

Stores per-dataset activations as a flat (total_tokens, d_model) bf16 memmap
with an index file tracking per-sample offsets, lengths, and metadata.

Cache layout::

    {cache_dir}/
        {dataset_name}/
            activations.bin    # flat bf16 memmap (total_tokens, d_model)
            index.pt           # sample metadata: offsets, lengths, labels, spans

Example:
    >>> cache = ActivationCache("./activation_cache", extractor, loader)
    >>> cache.precompute(["EnronDataset", "BIPIADataset"], config_entries)
    >>> ds = CachedActivationDataset(cache, ["EnronDataset", "BIPIADataset"])
    >>> batch = ds.load_batch(indices)
"""

import hashlib
import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm


class ActivationCache:
    """Manages per-dataset bf16 activation cache on disk.

    Args:
        cache_dir: Root directory for cache files (e.g., ./activation_cache).
        extractor: ActivationExtractor instance for computing activations.
        loader: DataLoader instance for loading dataset samples.
        batch_size: Batch size for extraction forward passes.
    """

    def __init__(self, cache_dir: str, extractor=None, loader=None, batch_size: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        self.loader = loader
        self.batch_size = batch_size

        meta_path = self.cache_dir / "meta.json"

        if extractor is not None:
            # Full mode: validate or create metadata
            self.d_model = extractor.d_model
            meta = {
                "model_name": extractor.model_name,
                "layer": extractor.layer,
                "max_seq_len": extractor.max_seq_len,
                "d_model": self.d_model,
            }
            if meta_path.exists():
                with open(meta_path) as f:
                    existing = json.load(f)
                if existing != meta:
                    raise ValueError(
                        f"Cache metadata mismatch.\n"
                        f"  Existing: {existing}\n"
                        f"  Current:  {meta}\n"
                        f"Delete {cache_dir} to rebuild."
                    )
            else:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        else:
            # Read-only mode: load metadata from disk
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"No meta.json in {cache_dir}. Cannot open cache without extractor."
                )
            with open(meta_path) as f:
                meta = json.load(f)
            self.d_model = meta["d_model"]

    def dataset_dir(self, dataset_name: str) -> Path:
        return self.cache_dir / dataset_name

    def is_cached(self, dataset_name: str) -> bool:
        """Check if a dataset has been fully cached."""
        d = self.dataset_dir(dataset_name)
        return (d / "activations.bin").exists() and (d / "index.pt").exists()

    def precompute_dataset(self, dataset_name: str, loader_spec) -> dict:
        """Extract and cache all activations for a single dataset.

        Args:
            dataset_name: Dataset class name.
            loader_spec: Spec passed to DataLoader.load().

        Returns:
            Dict with cache stats (n_samples, total_tokens, size_bytes, time_s).
        """
        if self.is_cached(dataset_name):
            index = torch.load(
                self.dataset_dir(dataset_name) / "index.pt", weights_only=True,
            )
            n = len(index["lengths"])
            total_tok = int(index["lengths"].sum().item())
            size = os.path.getsize(self.dataset_dir(dataset_name) / "activations.bin")
            print(f"  {dataset_name}: already cached ({n} samples, "
                  f"{total_tok} tokens, {size / 1e9:.2f} GB)")
            return {"n_samples": n, "total_tokens": total_tok,
                    "size_bytes": size, "time_s": 0, "skipped": True}

        print(f"  {dataset_name}: loading samples...")
        samples = self.loader.load(loader_spec)
        if not samples:
            print(f"    No samples loaded, skipping")
            return {"n_samples": 0, "total_tokens": 0,
                    "size_bytes": 0, "time_s": 0, "skipped": True}

        return self._extract_and_save(dataset_name, samples)

    def _extract_and_save(self, dataset_name: str, samples: list) -> dict:
        """Run extraction on samples and stream directly to memmap.

        Two-pass approach:
          1. Tokenize all prompts (CPU-only, fast) to get per-sample token counts.
          2. Allocate memmap of exact size, then stream GPU extractions into it
             using PrefetchExtractor for GPU/CPU overlap.
        """
        from activation_robustness.data.activation_extractor import PrefetchExtractor

        t0 = time.time()
        d = self.dataset_dir(dataset_name)
        d.mkdir(parents=True, exist_ok=True)

        prompts = [s["text"] for s in samples]
        n = len(prompts)
        bs = self.batch_size

        # --- Pass 1: tokenize to get per-sample lengths (CPU only, fast) ---
        print(f"    Pass 1: counting tokens...")
        lengths = torch.zeros(n, dtype=torch.int32)
        for start in range(0, n, bs):
            batch = prompts[start : start + bs]
            encoded = self.extractor.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.extractor.max_seq_len,
            )
            batch_lens = encoded["attention_mask"].sum(dim=1)
            lengths[start : start + len(batch)] = batch_lens.to(torch.int32)

        total_tokens = int(lengths.sum().item())
        offsets_arr = torch.zeros(n, dtype=torch.int64)
        offsets_arr[1:] = lengths[:-1].cumsum(0).to(torch.int64)
        print(f"    {n} samples, {total_tokens:,} tokens, "
              f"estimated {total_tokens * self.d_model * 2 / 1e9:.2f} GB")

        # --- Allocate memmap ---
        bin_path = d / "activations.bin"
        nbytes = total_tokens * self.d_model * 2
        storage = torch.UntypedStorage.from_file(str(bin_path), shared=True, nbytes=nbytes)
        mmap_tensor = torch.empty(0, dtype=torch.bfloat16).set_(storage).reshape(
            total_tokens, self.d_model,
        )

        # --- Pass 2: extract with GPU/CPU overlap, stream to memmap ---
        print(f"    Pass 2: extracting activations...")
        batch_indices = [
            list(range(start, min(start + bs, n)))
            for start in range(0, n, bs)
        ]
        prefetch = PrefetchExtractor(
            self.extractor, prompts, batch_indices, return_offsets=True,
        )

        per_sample_offsets = [None] * n
        for idx, (hidden, attn_mask, offset_maps) in tqdm(
            prefetch, desc=f"    Extracting {dataset_name}", total=len(batch_indices),
        ):
            # hidden: (B, T', D) bf16 GPU, attn_mask: (B, T') bool GPU
            hidden_cpu = hidden.cpu()
            mask_cpu = attn_mask.cpu()

            for j, sample_idx in enumerate(idx):
                m = mask_cpu[j]
                sample_hidden = hidden_cpu[j][m]  # (T_i, D)
                off = int(offsets_arr[sample_idx])
                length = int(lengths[sample_idx])
                mmap_tensor[off : off + length].copy_(sample_hidden)
                per_sample_offsets[sample_idx] = offset_maps[j][m]

            del hidden, attn_mask

        del mmap_tensor, storage

        # --- Build index ---
        labels = torch.tensor(
            [1.0 if s["labels"].get("malicious", False) else 0.0 for s in samples],
            dtype=torch.float32,
        )
        index = {
            "offsets": offsets_arr,
            "lengths": lengths,
            "labels": labels,
            "injection_spans": [s["injection_span"] for s in samples],
            "prompt_ids": [s["prompt_id"] for s in samples],
            "dataset_ids": [s["dataset_id"] for s in samples],
            "offset_mappings": per_sample_offsets,
            "n_samples": n,
            "total_tokens": total_tokens,
        }
        torch.save(index, d / "index.pt")

        elapsed = time.time() - t0
        size = os.path.getsize(bin_path)
        print(f"    Done: {size / 1e9:.2f} GB, {elapsed:.0f}s")

        return {"n_samples": n, "total_tokens": total_tokens,
                "size_bytes": size, "time_s": elapsed, "skipped": False}

    def precompute(self, entries, entry_to_spec_fn) -> dict:
        """Precompute activations for a list of dataset entries.

        Args:
            entries: List of DatasetEntry (str or dict).
            entry_to_spec_fn: Callable to convert entry to loader spec.

        Returns:
            Dict mapping dataset_name -> stats.
        """
        from activation_robustness.probes.config import _entry_class

        stats = {}
        for entry in entries:
            name = _entry_class(entry)
            spec = entry_to_spec_fn(entry)
            try:
                stats[name] = self.precompute_dataset(name, spec)
            except Exception as e:
                print(f"  ERROR caching {name}: {e}")
                stats[name] = {"n_samples": 0, "total_tokens": 0,
                               "size_bytes": 0, "time_s": 0, "error": str(e)}
        return stats


class CachedActivationDataset:
    """Random-access dataset backed by cached bf16 activations.

    Loads index files into RAM and memory-maps activation files for
    zero-copy reads. Supports shuffling via index permutation and
    efficient batch loading with dynamic padding.

    Args:
        cache: ActivationCache instance (for paths and d_model).
        dataset_names: List of dataset names to include.
    """

    def __init__(self, cache: ActivationCache, dataset_names: List[str]):
        self.d_model = cache.d_model
        self._mmaps: List[torch.Tensor] = []
        self._dataset_indices: List[int] = []  # which mmap each sample belongs to

        # Per-sample arrays (concatenated across datasets)
        all_offsets = []
        all_lengths = []
        all_labels = []
        all_spans = []
        all_offset_mappings = []
        all_prompt_ids = []
        all_dataset_ids = []

        for ds_name in dataset_names:
            ds_dir = cache.dataset_dir(ds_name)
            if not (ds_dir / "index.pt").exists():
                raise FileNotFoundError(
                    f"Cache not found for {ds_name}. Run with --precompute first."
                )

            index = torch.load(ds_dir / "index.pt", weights_only=True)
            n = index["n_samples"]
            total_tokens = index["total_tokens"]

            # Open memmap (read-only)
            nbytes = total_tokens * cache.d_model * 2
            storage = torch.UntypedStorage.from_file(
                str(ds_dir / "activations.bin"), shared=False, nbytes=nbytes,
            )
            mmap = torch.empty(0, dtype=torch.bfloat16).set_(storage).reshape(
                total_tokens, cache.d_model,
            )
            mmap_idx = len(self._mmaps)
            self._mmaps.append(mmap)

            self._dataset_indices.extend([mmap_idx] * n)
            all_offsets.append(index["offsets"])
            all_lengths.append(index["lengths"])
            all_labels.append(index["labels"])
            all_spans.extend(index["injection_spans"])
            all_offset_mappings.extend(index["offset_mappings"])
            all_prompt_ids.extend(index["prompt_ids"])
            all_dataset_ids.extend(index["dataset_ids"])

        self.offsets = torch.cat(all_offsets)
        self.lengths = torch.cat(all_lengths)
        self.labels = torch.cat(all_labels)
        self.injection_spans = all_spans
        self.offset_mappings = all_offset_mappings
        self.prompt_ids = all_prompt_ids
        self.dataset_ids = all_dataset_ids
        self._dataset_indices = torch.tensor(self._dataset_indices, dtype=torch.int32)

    def __len__(self):
        return len(self.labels)

    def load_batch(
        self,
        indices: Union[List[int], np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
        return_offsets: bool = False,
        max_tokens: Optional[int] = None,
    ) -> tuple:
        """Load a batch of samples by index, with left-padding to max length.

        Args:
            indices: Sample indices into this dataset.
            device: Target device for output tensors.
            return_offsets: If True, also return offset mappings.
            max_tokens: If set, only load the last N tokens per sample.
                Reduces memory and copy time for probes that only need
                positions near the end of the sequence.

        Returns:
            hidden: (B, T_max, D) bf16 tensor, left-padded.
            mask: (B, T_max) bool tensor.
            labels: (B,) float32 tensor.
            spans: List[Optional[Tuple]] of injection spans.
            offset_maps (optional): (B, T_max, 2) int tensor when return_offsets=True.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        batch_size = len(indices)
        batch_lengths = self.lengths[indices]

        if max_tokens is not None:
            # Clamp each sample's effective length to max_tokens
            effective_lengths = torch.clamp(batch_lengths, max=max_tokens)
        else:
            effective_lengths = batch_lengths

        max_len = int(effective_lengths.max().item())

        hidden = torch.zeros(
            batch_size, max_len, self.d_model, dtype=torch.bfloat16,
        )
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        if return_offsets:
            offset_maps = torch.zeros(batch_size, max_len, 2, dtype=torch.int64)

        for j, sample_idx in enumerate(indices):
            ds_idx = int(self._dataset_indices[sample_idx])
            off = int(self.offsets[sample_idx])
            full_length = int(self.lengths[sample_idx])
            eff_length = int(effective_lengths[j])

            # Read only the last eff_length tokens from the sample
            read_off = off + (full_length - eff_length)

            # Left-pad: place at right end
            start = max_len - eff_length
            hidden[j, start:] = self._mmaps[ds_idx][read_off : read_off + eff_length]
            mask[j, start:] = True

            if return_offsets:
                om = self.offset_mappings[sample_idx]
                if isinstance(om, torch.Tensor):
                    om = om[full_length - eff_length:]
                    offset_maps[j, start:] = om
                else:
                    om_t = torch.tensor(om, dtype=torch.int64)
                    offset_maps[j, start:] = om_t[full_length - eff_length:]

        batch_labels = self.labels[indices]
        batch_spans = [self.injection_spans[i] for i in indices]

        if device is not None:
            hidden = hidden.to(device)
            mask = mask.to(device)
            batch_labels = batch_labels.to(device)
            if return_offsets:
                offset_maps = offset_maps.to(device)

        if return_offsets:
            return hidden, mask, batch_labels, batch_spans, offset_maps
        return hidden, mask, batch_labels, batch_spans

    def make_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Create batch index arrays, optionally shuffled.

        Args:
            batch_size: Samples per batch.
            shuffle: Whether to shuffle sample order.
            seed: Random seed for reproducibility.

        Returns:
            List of index arrays, one per batch.
        """
        n = len(self)
        if shuffle:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
        else:
            perm = np.arange(n)

        return [
            perm[start : start + batch_size]
            for start in range(0, n, batch_size)
        ]
