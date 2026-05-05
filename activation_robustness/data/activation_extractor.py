"""
Activation extractor for LLM hidden states.

Loads a HuggingFace causal LM once in bf16 and provides batched extraction
of hidden states at a specified layer. Supports:
- All-position extraction (for token-level probes)
- Position-specific extraction (for sequence-level probes)
- Async tokenization for CPU/GPU overlap via PrefetchExtractor

Example:
    >>> ext = ActivationExtractor("meta-llama/Llama-3.1-8B-Instruct", layer=31)
    >>> hidden, mask = ext.extract_all_positions(prompts)
    >>> X = ext.extract(prompts, position="last")  # (n, d_model)
"""

import queue
import threading
import numpy as np
import torch
from typing import Optional, Union, List, Callable
from tqdm import tqdm


def _pool_with_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    pool: str,
) -> torch.Tensor:
    """Pool (B, N, D) tensor over dim 1 respecting mask (B, N)."""
    if x.shape[1] == 1:
        return x[:, 0]
    if pool == "first":
        return x[:, 0]
    elif pool == "mean":
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    elif pool == "max":
        x = x.clone()
        x[~mask] = float("-inf")
        return x.max(dim=1).values
    else:
        raise ValueError(f"Unknown pool mode: {pool}")


class ActivationExtractor:
    """Extracts hidden-state activations from an LLM.

    Loads the model once in bf16 and keeps it on GPU. Extraction returns
    the target layer's hidden states.

    Args:
        model_name: HuggingFace model ID.
        layer: Transformer layer index (0-based).
        device: PyTorch device. None = auto cuda.
        max_seq_len: Truncate prompts longer than this.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        layer: int = 31,
        device: Optional[str] = None,
        max_seq_len: int = 16384,
        attn_implementation: Optional[str] = "sdpa",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.layer = layer
        self.max_seq_len = max_seq_len
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs,
        )
        self.model.eval()

        # hidden_states index: [0]=embeddings, [i+1]=output of layer i
        self._hs_index = layer + 1
        self._d_model = self.model.config.hidden_size

    @property
    def d_model(self) -> int:
        return self._d_model

    # ----- Core extraction methods -----

    def extract_all_positions(
        self,
        prompts: List[str],
        return_offsets: bool = False,
    ) -> tuple:
        """Extract all-position hidden states for a batch, trimming left padding.

        Args:
            prompts: List of prompt strings.
            return_offsets: If True, also return tokenizer offset mappings
                (needed for character-span to token-label conversion).

        Returns:
            hidden: (B, T', D) bf16 tensor on GPU, where T' = max actual
                length in the batch. Shorter sequences have padding at left.
            attn_mask: (B, T') bool tensor on GPU. True for real tokens.
            offsets (optional): (B, T', 2) tensor of character offsets when
                return_offsets=True.
        """
        tokenizer_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        if return_offsets:
            tokenizer_kwargs["return_offsets_mapping"] = True

        encoded = self.tokenizer(prompts, **tokenizer_kwargs)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[self._hs_index]
            del out

        # Trim left padding: keep rightmost max_valid tokens
        max_valid = int(attention_mask.sum(dim=1).max().item())
        hidden_trimmed = hidden[:, -max_valid:].clone()
        mask_trimmed = attention_mask[:, -max_valid:].bool()

        del hidden, input_ids, attention_mask

        if return_offsets:
            offsets = encoded["offset_mapping"][:, -max_valid:]
            return hidden_trimmed, mask_trimmed, offsets

        return hidden_trimmed, mask_trimmed

    # ----- Async tokenization for CPU/GPU overlap -----

    def tokenize_batch_async(
        self,
        prompts: List[str],
        return_offsets: bool = False,
    ) -> dict:
        """Tokenize a batch on CPU, returning tensors NOT yet on GPU.

        Used by PrefetchExtractor to overlap tokenization with GPU work.
        """
        tokenizer_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        if return_offsets:
            tokenizer_kwargs["return_offsets_mapping"] = True
        return self.tokenizer(prompts, **tokenizer_kwargs)

    def forward_from_encoded(
        self,
        encoded: dict,
        return_offsets: bool = False,
    ) -> tuple:
        """Run model forward from pre-tokenized batch.

        Moves tensors to GPU, runs forward pass, trims padding.
        Same return signature as extract_all_positions.
        """
        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[self._hs_index]
            del out

        max_valid = int(attention_mask.sum(dim=1).max().item())
        hidden_trimmed = hidden[:, -max_valid:].clone()
        mask_trimmed = attention_mask[:, -max_valid:].bool()

        del hidden, input_ids, attention_mask

        if return_offsets:
            offsets = encoded["offset_mapping"][:, -max_valid:]
            return hidden_trimmed, mask_trimmed, offsets

        return hidden_trimmed, mask_trimmed

    # ----- Position-specific extraction -----

    def _tokenize_batch(self, prompts: List[str]):
        """Tokenize and left-pad a batch of prompts.

        Returns:
            input_ids: (batch, max_len) on GPU
            attention_mask: (batch, max_len) on GPU
            seq_lens: list of actual lengths per sequence
        """
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        seq_lens = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, seq_lens

    def _resolve_positions_for_sequence(
        self,
        input_ids_seq: torch.Tensor,
        seq_len: int,
        padded_len: int,
        position_spec: Union[str, List[int], Callable],
    ) -> List[int]:
        """Resolve position spec to absolute indices into the padded sequence."""
        pad_offset = padded_len - seq_len

        if callable(position_spec) and not isinstance(position_spec, str):
            unpadded_ids = input_ids_seq[pad_offset:].tolist()
            raw_positions = position_spec(unpadded_ids)
            return [p + pad_offset for p in raw_positions]

        if isinstance(position_spec, str):
            if position_spec == "last":
                return [padded_len - 1]
            elif position_spec.startswith("token:"):
                token_str = position_spec[len("token:"):]
                token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                if token_id == self.tokenizer.unk_token_id:
                    token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                    if len(token_ids) == 1:
                        token_id = token_ids[0]
                    else:
                        raise ValueError(
                            f"Token string '{token_str}' encodes to multiple tokens: {token_ids}. "
                            f"Use a single-token string."
                        )
                unpadded = input_ids_seq[pad_offset:]
                matches = (unpadded == token_id).nonzero(as_tuple=True)[0].tolist()
                if not matches:
                    return [padded_len - 1]
                return [m + pad_offset for m in matches]
            else:
                raise ValueError(f"Unknown position spec string: {position_spec}")

        if isinstance(position_spec, list):
            result = []
            for p in position_spec:
                if p >= 0:
                    result.append(p + pad_offset)
                else:
                    result.append(padded_len + p)
            return result

        raise ValueError(f"Invalid position spec: {position_spec}")

    def extract_batch(
        self,
        prompts: List[str],
        position: Union[str, List[int], Callable] = "last",
    ) -> tuple:
        """Extract activations for a batch of prompts at specific positions.

        Args:
            prompts: List of prompt strings.
            position: Position specification (e.g. "last", [-5], "token:<tok>").

        Returns:
            acts: (batch, max_n_positions, d_model) bf16 tensor on GPU
            mask: (batch, max_n_positions) bool tensor — True for valid positions
        """
        input_ids, attention_mask, seq_lens = self._tokenize_batch(prompts)
        batch_size, padded_len = input_ids.shape

        with torch.inference_mode():
            out = self.model(
                input_ids, attention_mask=attention_mask,
                use_cache=False, output_hidden_states=True,
            )
            hidden = out.hidden_states[self._hs_index]
            del out

        all_positions = []
        for i in range(batch_size):
            pos = self._resolve_positions_for_sequence(
                input_ids[i], int(seq_lens[i]), padded_len, position
            )
            all_positions.append(pos)

        max_n_pos = max(len(p) for p in all_positions)

        acts = torch.zeros(
            batch_size, max_n_pos, self._d_model,
            dtype=torch.bfloat16, device=self.device,
        )
        mask = torch.zeros(batch_size, max_n_pos, dtype=torch.bool, device=self.device)

        for i, pos_list in enumerate(all_positions):
            for j, p in enumerate(pos_list):
                acts[i, j] = hidden[i, p]
                mask[i, j] = True

        del hidden
        return acts, mask

    def extract(
        self,
        prompts: List[str],
        position: Union[str, List[int], Callable] = "last",
        batch_size: int = 64,
        pool: str = "mean",
        verbose: bool = True,
    ) -> np.ndarray:
        """Extract activations for many prompts, returning numpy.

        Convenience method that batches, pools, and returns CPU numpy.

        Args:
            prompts: All prompts.
            position: Position spec.
            batch_size: Prompts per GPU batch.
            pool: Pooling over positions: "first", "mean", "max".
            verbose: Show progress bar.

        Returns:
            (n_prompts, d_model) float32 numpy array.
        """
        all_acts = []
        iterator = range(0, len(prompts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Extracting activations")

        for start in iterator:
            batch = prompts[start : start + batch_size]
            acts, mask = self.extract_batch(batch, position)
            pooled = _pool_with_mask(acts, mask, pool)
            all_acts.append(pooled.float().cpu().numpy())

        return np.concatenate(all_acts, axis=0)


class PrefetchExtractor:
    """Iterator that prefetches tokenized batches on a background thread.

    While the GPU processes batch N (model forward + probe + backward),
    the CPU tokenizes batch N+1 in parallel. On H100 with long sequences
    this hides most of the tokenization latency.

    Requires an ActivationExtractor with ``tokenize_batch_async`` and
    ``forward_from_encoded`` methods.

    Args:
        extractor: ActivationExtractor instance.
        prompts: Full list of prompt strings.
        batch_indices: List of index arrays (one per batch).
        return_offsets: Whether to return offset mappings.
        prefetch_depth: How many batches to tokenize ahead (1 is enough).
    """

    def __init__(
        self,
        extractor,
        prompts: List[str],
        batch_indices: List,
        return_offsets: bool = False,
        prefetch_depth: int = 1,
    ):
        self._extractor = extractor
        self._prompts = prompts
        self._batch_indices = batch_indices
        self._return_offsets = return_offsets
        self._prefetch_depth = prefetch_depth

    def __len__(self):
        return len(self._batch_indices)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self._prefetch_depth)
        sentinel = object()

        def _producer():
            for idx in self._batch_indices:
                batch_prompts = [self._prompts[i] for i in idx]
                encoded = self._extractor.tokenize_batch_async(
                    batch_prompts, return_offsets=self._return_offsets,
                )
                q.put((idx, encoded))
            q.put(sentinel)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            idx, encoded = item
            result = self._extractor.forward_from_encoded(
                encoded, return_offsets=self._return_offsets,
            )
            yield idx, result

        thread.join()
