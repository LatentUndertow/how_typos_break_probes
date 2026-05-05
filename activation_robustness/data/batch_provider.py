"""
Batch providers for probe training and prediction.

Abstracts whether activations come from live LLM extraction or cached
bf16 memmaps. Probes iterate over a provider without knowing the source.

Both providers yield the same tuple format:
    Training:  (hidden, mask, labels, spans[, offset_maps])
    Predict:   (hidden, mask)

Usage::

    # In run_experiment.py — pick provider based on cache availability
    if cache_exists:
        provider = CachedBatchProvider(cached_dataset)
    else:
        provider = ExtractorBatchProvider(extractor, prompts, labels, spans)

    # In probe — identical loop regardless of source
    for hidden, mask, labels, spans, offset_maps in provider.iter_train_batches(
        batch_size=6, epoch_seed=42, device=device, return_offsets=True,
    ):
        ...
"""

import abc
import queue
import threading

import numpy as np
import torch
from typing import Iterator, Optional, List, Tuple, Union


_SENTINEL = object()  # signals end of iteration


class BatchProvider(abc.ABC):
    """Abstract interface for batch providers.

    All batch providers expose the same attributes and iteration methods
    so that classifiers can consume training/prediction batches without
    knowing whether activations come from a cache or live extraction.
    """

    n_samples: int
    d_model: int

    @property
    @abc.abstractmethod
    def labels(self) -> np.ndarray:
        """Binary labels (n_samples,) as float32 numpy array."""
        ...

    @abc.abstractmethod
    def iter_train_batches(
        self,
        batch_size: int,
        epoch_seed: int = 0,
        device: Optional[torch.device] = None,
        return_offsets: bool = False,
    ) -> Iterator:
        """Yield shuffled training batches."""
        ...

    @abc.abstractmethod
    def iter_predict_batches(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Iterator:
        """Yield sequential prediction batches as (hidden, mask)."""
        ...


class _PrefetchIterator:
    """Multi-worker prefetch wrapper for CachedActivationDataset.load_batch.

    Spawns *num_workers* daemon threads that load batches from the NVMe mmap
    in parallel, feeding results through a bounded queue so the consumer
    (GPU training/inference) always has a batch ready.

    Pipeline (num_workers=4)::

        Worker 0: [load batch 0] [load batch 4] [load batch 8]  ...
        Worker 1: [load batch 1] [load batch 5] [load batch 9]  ...
        Worker 2: [load batch 2] [load batch 6] [load batch 10] ...
        Worker 3: [load batch 3] [load batch 7] [load batch 11] ...
        Main:              [yield 0] [yield 1] [yield 2] [yield 3] ...

    An ordering queue ensures batches are yielded in the original sequence
    even though workers may finish out of order.

    Args:
        dataset: CachedActivationDataset instance.
        batch_indices: List of index arrays (one per batch).
        device: Target device for output tensors.
        load_kwargs: Extra kwargs forwarded to dataset.load_batch.
        num_workers: Number of parallel loader threads. Default 4.
        queue_depth: Number of batches to buffer ahead. Default num_workers*2.
        transform_fn: Optional callable applied to each load_batch result.
    """

    def __init__(
        self,
        dataset,
        batch_indices: list,
        device: Optional[torch.device] = None,
        load_kwargs: Optional[dict] = None,
        num_workers: int = 4,
        queue_depth: Optional[int] = None,
        transform_fn=None,
    ):
        self._dataset = dataset
        self._batch_indices = batch_indices
        self._device = device
        self._load_kwargs = load_kwargs or {}
        self._transform_fn = transform_fn
        self._num_workers = max(1, num_workers)

        if queue_depth is None:
            queue_depth = self._num_workers * 2
        self._n_batches = len(batch_indices)

        # Work queue: (sequence_index, batch_idx) pairs for workers to pick up
        self._work_queue: queue.Queue = queue.Queue()
        for seq_i, batch_idx in enumerate(batch_indices):
            self._work_queue.put((seq_i, batch_idx))

        # Results dict + lock for out-of-order completion
        self._results: dict = {}
        self._results_lock = threading.Lock()
        self._results_ready = threading.Condition(self._results_lock)

        # Semaphore limits how far ahead workers can get (backpressure)
        self._slots = threading.Semaphore(queue_depth)

        self._errors: list = []
        self._next_seq = 0  # next sequence index to yield

        self._workers = []
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker, daemon=True, name=f"prefetch-worker-{i}",
            )
            t.start()
            self._workers.append(t)

    def _to_device(self, batch_tuple):
        """Move torch.Tensor elements to the target device."""
        if self._device is None:
            return batch_tuple
        result = []
        for item in batch_tuple:
            if isinstance(item, torch.Tensor):
                result.append(item.to(self._device, non_blocking=False))
            else:
                result.append(item)
        return tuple(result)

    def _worker(self):
        """Worker thread: pull (seq_index, batch_idx) from work queue, load, store."""
        while True:
            try:
                seq_i, batch_idx = self._work_queue.get_nowait()
            except queue.Empty:
                return  # no more work

            # Backpressure: wait if consumer is behind
            self._slots.acquire()

            try:
                batch = self._dataset.load_batch(batch_idx, **self._load_kwargs)
                if self._transform_fn is not None:
                    batch = self._transform_fn(batch)
            except BaseException as exc:
                with self._results_lock:
                    self._errors.append(exc)
                    self._results_ready.notify_all()
                return

            with self._results_lock:
                self._results[seq_i] = batch
                self._results_ready.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_seq >= self._n_batches:
            raise StopIteration

        # Wait for the next in-order batch
        with self._results_lock:
            while self._next_seq not in self._results:
                if self._errors:
                    raise self._errors[0]
                self._results_ready.wait(timeout=1.0)

            batch = self._results.pop(self._next_seq)

        self._next_seq += 1
        self._slots.release()  # free a slot for workers to fill

        return self._to_device(batch)


class CachedBatchProvider(BatchProvider):
    """Batch provider backed by CachedActivationDataset (NVMe memmap).

    Args:
        cached_dataset: CachedActivationDataset instance.
        indices: Optional array of sample indices to restrict to. When
            provided, only these indices are served (for train/test splits).
            Indices refer to positions in the underlying dataset.
        num_workers: Number of parallel loader threads for prefetch.
            More workers overlap more mmap page faults. Default 4.
    """

    def __init__(self, cached_dataset, indices: Optional[np.ndarray] = None,
                 num_workers: int = 4, max_tokens: Optional[int] = None):
        self._ds = cached_dataset
        self._num_workers = num_workers
        self._indices = indices  # None = use all samples
        self._max_tokens = max_tokens  # None = load full sequences
        if indices is not None:
            self.n_samples = len(indices)
        else:
            self.n_samples = len(cached_dataset)
        self.d_model = cached_dataset.d_model

    @property
    def labels(self) -> np.ndarray:
        """Binary labels (n_samples,) as float32 numpy array."""
        all_labels = self._ds.labels.numpy().astype(np.float32)
        if self._indices is not None:
            return all_labels[self._indices]
        return all_labels

    def _make_batches(self, batch_size: int, shuffle: bool = True,
                      seed: Optional[int] = None) -> List[np.ndarray]:
        """Create batch index arrays over this provider's subset."""
        if self._indices is not None:
            # Shuffle/order within our subset, but return dataset-level indices
            n = len(self._indices)
            if shuffle:
                rng = np.random.RandomState(seed)
                perm = rng.permutation(n)
            else:
                perm = np.arange(n)
            subset = self._indices[perm]
            return [
                subset[start : start + batch_size]
                for start in range(0, n, batch_size)
            ]
        return self._ds.make_batches(batch_size, shuffle=shuffle, seed=seed)

    def iter_train_batches(
        self,
        batch_size: int,
        epoch_seed: int = 0,
        device: Optional[torch.device] = None,
        return_offsets: bool = False,
    ):
        """Yield shuffled training batches from cache with background prefetch.

        A background thread loads the next batch from the NVMe mmap while
        the current batch is being processed on GPU, overlapping CPU I/O
        with GPU computation.

        Yields:
            If return_offsets: (hidden, mask, labels, spans, offset_maps)
            Else: (hidden, mask, labels, spans)
        """
        batches = self._make_batches(batch_size, shuffle=True, seed=epoch_seed)
        prefetch = _PrefetchIterator(
            dataset=self._ds,
            batch_indices=batches,
            device=device,
            load_kwargs={"return_offsets": return_offsets, "max_tokens": self._max_tokens},
            num_workers=self._num_workers,
        )
        yield from prefetch

    def iter_predict_batches(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ):
        """Yield sequential prediction batches with background prefetch.

        Yields:
            (hidden, mask) tuples in sample order.
        """
        batches = self._make_batches(batch_size, shuffle=False)

        def _extract_hidden_mask(batch):
            """Strip labels and spans, keep only (hidden, mask)."""
            hidden, mask, _labels, _spans = batch
            return (hidden, mask)

        prefetch = _PrefetchIterator(
            dataset=self._ds,
            batch_indices=batches,
            device=device,
            load_kwargs={"return_offsets": False, "max_tokens": self._max_tokens},
            num_workers=self._num_workers,
            transform_fn=_extract_hidden_mask,
        )
        yield from prefetch


class ExtractorBatchProvider(BatchProvider):
    """Batch provider using live LLM activation extraction.

    Wraps PrefetchExtractor to provide the same interface as CachedBatchProvider.

    Args:
        extractor: ActivationExtractor instance.
        prompts: List of prompt strings.
        labels: Binary labels array (n,). Can be None for predict-only.
        injection_spans: Per-prompt (start_char, end_char) or None.
    """

    def __init__(
        self,
        extractor,
        prompts: List[str],
        labels=None,
        injection_spans: Optional[List] = None,
    ):
        self.extractor = extractor
        self.prompts = prompts
        self._labels_np = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self._spans = injection_spans
        self.n_samples = len(prompts)
        self.d_model = extractor.d_model

    @property
    def labels(self) -> np.ndarray:
        """Binary labels (n_samples,) as float32 numpy array."""
        if self._labels_np is None:
            raise ValueError("No labels available (predict-only provider).")
        return self._labels_np

    def iter_train_batches(
        self,
        batch_size: int,
        epoch_seed: int = 0,
        device: Optional[torch.device] = None,
        return_offsets: bool = False,
    ):
        """Yield shuffled training batches via live LLM extraction.

        Uses PrefetchExtractor to overlap CPU tokenization with GPU work.

        Yields:
            If return_offsets: (hidden, mask, labels, spans, offset_maps)
            Else: (hidden, mask, labels, spans)
        """
        from activation_robustness.data.activation_extractor import PrefetchExtractor

        n = self.n_samples
        rng = np.random.RandomState(epoch_seed)
        perm = rng.permutation(n)

        batch_indices = [
            perm[start: start + batch_size]
            for start in range(0, n, batch_size)
        ]

        prefetch = PrefetchExtractor(
            self.extractor, self.prompts, batch_indices,
            return_offsets=return_offsets,
        )

        labels_t = torch.tensor(self._labels_np, dtype=torch.float32)

        for idx, result in prefetch:
            idx_list = list(idx) if not isinstance(idx, list) else idx
            batch_labels = labels_t[idx_list]
            if device is not None:
                batch_labels = batch_labels.to(device)
            batch_spans = (
                [self._spans[i] for i in idx_list]
                if self._spans is not None
                else [None] * len(idx_list)
            )

            if return_offsets:
                hidden, mask, offset_maps = result
                yield hidden, mask, batch_labels, batch_spans, offset_maps
            else:
                hidden, mask = result
                yield hidden, mask, batch_labels, batch_spans

    def iter_predict_batches(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ):
        """Yield sequential prediction batches via live LLM extraction.

        Yields:
            (hidden, mask) tuples in sample order.
        """
        from activation_robustness.data.activation_extractor import PrefetchExtractor

        n = self.n_samples
        batch_indices = [
            list(range(start, min(start + batch_size, n)))
            for start in range(0, n, batch_size)
        ]

        prefetch = PrefetchExtractor(
            self.extractor, self.prompts, batch_indices,
            return_offsets=False,
        )

        for _idx, (hidden, mask) in prefetch:
            yield hidden, mask
