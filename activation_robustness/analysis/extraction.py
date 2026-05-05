"""Model loading and activation extraction for HuggingFace models.

Provides functions for loading models, formatting prompts, and extracting
activations at specified layers and token positions. All activations are
returned as float64 numpy arrays for numerical stability in downstream
statistical analysis.
"""

from typing import Optional

import numpy as np
import torch


def load_hf_model(
    model_name: str,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
):
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier
            (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        dtype: Torch dtype string for model weights. One of
            "bfloat16", "float16", "float32". Default "bfloat16".
        device: Device to load onto. If None, uses "cuda" when available,
            otherwise "cpu".

    Returns:
        Tuple of (model, tokenizer). The model is in eval mode with
        output_hidden_states=True.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of {list(dtype_map)}")

    torch_dtype = dtype_map[dtype]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()

    return model, tokenizer


def format_prompt(
    tokenizer,
    user_text: str,
    steering_prompt: str = "",
    add_generation_prompt: bool = False,
) -> str:
    """Apply chat template with optional steering/system prompt.

    Args:
        tokenizer: HuggingFace tokenizer with a chat template.
        user_text: The user's message content.
        steering_prompt: System prompt or steering instruction. If empty,
            no system message is added.
        add_generation_prompt: Whether to append the assistant turn prefix
            (useful for generation, not needed for activation extraction
            at the last user-turn token).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    messages = []
    if steering_prompt:
        messages.append({"role": "system", "content": steering_prompt})
    messages.append({"role": "user", "content": user_text})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def extract_activations_hf(
    model,
    tokenizer,
    texts: list[str],
    layers: Optional[list[int]] = None,
    positions: Optional[list[int]] = None,
    batch_size: int = 8,
) -> dict[tuple[int, int], np.ndarray]:
    """Extract hidden-state activations for multiple texts.

    Processes texts in batches and collects activations at the specified
    layer and token-position combinations.

    Args:
        model: HuggingFace causal LM (with output_hidden_states=True).
        tokenizer: Corresponding HuggingFace tokenizer.
        texts: List of pre-formatted prompt strings (already passed through
            format_prompt or apply_chat_template).
        layers: List of transformer block indices to extract from.
            Uses hook_resid_post convention: layer L returns hidden_states[L+1],
            i.e., the output AFTER transformer block L.
            Default: [31] (last block). Pass None for all layers.
        positions: List of token positions to extract. Negative indices
            are supported (e.g., -1 for last token). Default: [-1].
            Pass None for all positions.
        batch_size: Number of texts per forward pass. Default 8.

    Returns:
        Dict mapping (layer_idx, position_idx) to a float64 numpy array
        of shape (n_texts, d_model). Position indices in the keys are
        the original values passed in (e.g., -1 stays as -1).
    """
    if layers is None:
        # Will be resolved after first forward pass
        _all_layers = True
        layers = []
    else:
        _all_layers = False

    if positions is None:
        _all_positions = True
        positions = []
    else:
        _all_positions = False

    if not _all_layers and not layers:
        layers = [31]
    if not _all_positions and not positions:
        positions = [-1]

    # Collect results per (layer, position) as lists of arrays
    results: dict[tuple[int, int], list[np.ndarray]] = {}

    device = next(model.parameters()).device

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, d_model)

        # Resolve "all layers" on first batch
        # hidden_states has n_layers+1 entries (0=embedding, 1..n=blocks)
        # We use block indices 0..n-1, accessing hidden_states[idx+1]
        if _all_layers and not layers:
            layers = list(range(len(hidden_states) - 1))  # exclude embedding

        # Determine actual sequence lengths (excluding padding) per sample
        attention_mask = encodings["attention_mask"]  # (batch, seq_len)
        seq_lengths = attention_mask.sum(dim=1).cpu()  # (batch,)

        for layer_idx in layers:
            # hook_resid_post: output AFTER block = hidden_states[layer_idx + 1]
            layer_hidden = hidden_states[layer_idx + 1]  # (batch, seq_len, d_model)

            # Resolve "all positions" on first batch
            if _all_positions and not positions:
                # Use positions relative to actual sequence length of first sample
                max_len = int(seq_lengths[0].item())
                positions = list(range(max_len))

            for pos in positions:
                key = (layer_idx, pos)
                if key not in results:
                    results[key] = []

                batch_acts = []
                for sample_i in range(len(batch_texts)):
                    seq_len = int(seq_lengths[sample_i].item())

                    # Resolve position relative to actual (non-padded) length
                    if pos < 0:
                        actual_pos = seq_len + pos
                    else:
                        actual_pos = pos

                    if actual_pos < 0 or actual_pos >= seq_len:
                        raise IndexError(
                            f"Position {pos} out of range for text {batch_start + sample_i} "
                            f"with sequence length {seq_len}"
                        )

                    # For left-padded models, offset by the padding amount
                    pad_offset = layer_hidden.shape[1] - seq_len
                    act = layer_hidden[sample_i, pad_offset + actual_pos, :]
                    batch_acts.append(act.cpu().float().numpy().astype(np.float64))

                results[key].append(np.stack(batch_acts, axis=0))

    # Concatenate batches
    return {
        key: np.concatenate(arrays, axis=0)
        for key, arrays in results.items()
    }
