"""Load and sample from external datasets.

Standalone-release version: dataset access goes directly through
HuggingFace ``datasets`` rather than through internal wrapper classes.
Sampling logic, RNG draws, scout sizes, and bucket filters are
unchanged from the original — only the dataset loader differs.
"""
from typing import Optional

import numpy as np


def sample_openorca(
    n_per_bucket: int = 50,
    buckets: Optional[dict[str, tuple[int, int]]] = None,
    scout_size: int = 20000,
    seed: int = 42,
) -> list[dict]:
    """Sample prompts from OpenOrca stratified by word count.

    Uses HF dataset .select() for efficient random access — does NOT
    iterate the full 4.2M dataset.

    Args:
        n_per_bucket: Samples per length bucket.
        buckets: Dict of {name: (min_words, max_words)}.
            Default covers 20-100, 100-200, 200-500, 500-1000, 1000+.
        scout_size: Random indices to scout for length distribution.
        seed: Random seed.

    Returns:
        List of dicts with keys: text, word_count, length_bucket, source, safety.
    """
    from datasets import load_dataset

    if buckets is None:
        buckets = {
            "20-100": (20, 100),
            "100-200": (100, 200),
            "200-500": (200, 500),
            "500-1000": (500, 1000),
            "1000+": (1000, 999999),
        }

    hf = load_dataset("Open-Orca/OpenOrca", split="train")
    total = len(hf)

    rng = np.random.default_rng(seed)
    scout_indices = rng.choice(total, size=min(scout_size, total), replace=False)
    scout = hf.select(scout_indices)

    # Compute word counts for scout sample
    scout_wcs = []
    for row in scout:
        text = row.get("question", "") or ""
        if row.get("system_prompt"):
            text = row["system_prompt"] + " " + text
        scout_wcs.append(len(text.split()))
    scout_wcs = np.array(scout_wcs)

    # Sample per bucket
    results = []
    for bucket_name, (lo, hi) in buckets.items():
        mask = (scout_wcs >= lo) & (scout_wcs < hi)
        available = scout_indices[mask]
        n = min(n_per_bucket, len(available))
        if n == 0:
            continue
        chosen = rng.choice(available, size=n, replace=False)
        rows = hf.select(chosen)
        for row in rows:
            text = row.get("question", "") or ""
            if row.get("system_prompt"):
                text = row["system_prompt"] + " " + text
            wc = len(text.split())
            results.append({
                "text": text,
                "word_count": wc,
                "length_bucket": bucket_name,
                "source": "openorca",
                "safety": "benign",
            })

    return results


def sample_malicious(
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Sample malicious prompts from HarmBench and AdvBench.

    Standalone-release version: pulls directly from public HF datasets.

    Args:
        n: Total malicious prompts to sample.
        seed: Random seed.

    Returns:
        List of dicts with keys: text, word_count, source, safety.
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    results = []

    sources = [
        # (name, hf_id, split, text_field)
        ("harmbench", "walledai/HarmBench", "train", "prompt"),
        ("advbench", "walledai/AdvBench", "train", "prompt"),
    ]
    for name, hf_id, split, text_field in sources:
        try:
            ds = load_dataset(hf_id, split=split)
            for row in ds:
                text = row.get(text_field, "") or ""
                if not text:
                    continue
                results.append({
                    "text": text,
                    "word_count": len(text.split()),
                    "source": name,
                    "safety": "malicious",
                })
        except Exception as e:
            print(f"Warning: could not load {name} ({hf_id}): {e}")

    # Sample n from combined
    if len(results) > n:
        indices = rng.choice(len(results), size=n, replace=False)
        results = [results[i] for i in indices]

    return results
