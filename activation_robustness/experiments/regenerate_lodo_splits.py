"""Regenerate LODO split_indices.npz files using canonical DATASETS_ALL ordering.

Writes new splits to lodo_sweep_fixed/ (does NOT overwrite original lodo_sweep/).
Validates each new fold: test contains only held-out dataset, train contains all
others, no train/test overlap, all indices in valid range.
"""
import sys
from pathlib import Path

import numpy as np



from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset

DATASETS_ALL = [
    "EnronDataset", "Dolly15kDataset", "OpenOrcaDataset", "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    "BIPIADataset", "InjecAgentDataset", "LLMailDataset", "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset", "HarmBenchDataset", "AgentDojoDataset",
    "APIGenMTDataset", "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset", "ScamDataset",
    "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

DATASET_TO_PREFIX = {
    "EnronDataset": "enron",
    "Dolly15kDataset": "dolly_15k",
    "OpenOrcaDataset": "openorca",
    "PromptsRanked10kDataset": "10k_prompts_ranked",
    "AlpacaDataset": "alpaca",
    "SafeGuardDataset": "safeguard",
    "QualifireDataset": "qualifire",
    "SoftAgeDataset": "softAge",
    "BIPIADataset": "bipia_email_code_table",
    "InjecAgentDataset": "injecagent_dh_ds_base",
    "LLMailDataset": "llmail",
    "MosscapDataset": "mosscap",
    "WildJailbreakDataset": "wildjailbreak",
    "JayavibhavDataset": "jayavibhav",
    "DeepsetDataset": "deepset",
    "YanismiraouiDataset": "yanismiraoui",
    "AdvBenchDataset": "advbench",
    "HarmBenchDataset": "harmbench",
    "AgentDojoDataset": "agentdojo",
    "APIGenMTDataset": "apigen_mt",
    "BitextCustomerSupportDataset": "bitext_customer_support",
    "CodeExerciseDataset": "code_exercise",
    "GandalfSummarizationDataset": "gandalf_summarization",
    "JailbreakClassificationDataset": "jailbreak_classification",
    "PythonCodeAlpacaDataset": "python_code_alpaca",
    "PythonCodes25kDataset": "python_codes_25k",
    "ScamDataset": "scam",
    "WritingPromptsDataset": "writingprompts",
    "XlamFunctionCallingDataset": "xlam_function_calling",
}

SRC_DIR  = Path("./interpretability-research/activation_robustness/results/lodo_sweep")
DEST_DIR = Path("./interpretability-research/activation_robustness/results/lodo_sweep_fixed")


def main():
    cache = ActivationCache(cache_dir=os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations"))
    ds = CachedActivationDataset(cache, DATASETS_ALL)
    pids = np.array(ds.prompt_ids)
    n_total = len(pids)
    print(f"Loaded {n_total} samples from canonical DATASETS_ALL ordering")

    # Sanity: every sample's prefix should be one of our known prefixes
    known_prefixes = set(DATASET_TO_PREFIX.values())
    actual_prefixes = set(p.split(":")[0] for p in pids)
    unknown = actual_prefixes - known_prefixes
    if unknown:
        raise RuntimeError(f"Unknown prefixes in cache: {unknown}")
    extras = known_prefixes - actual_prefixes
    if extras:
        print(f"  WARN: prefixes in mapping not in cache: {extras}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing new splits to: {DEST_DIR}")

    all_idx = np.arange(n_total)
    written = 0
    failed = []

    for ds_name in DATASETS_ALL:
        prefix = DATASET_TO_PREFIX[ds_name]
        # Only regenerate folds that exist in the source (so we mirror coverage)
        src_fold = SRC_DIR / f"fold_{ds_name}"
        if not src_fold.exists():
            print(f"  SKIP {ds_name}: no source fold")
            continue

        test_mask  = np.array([p.split(":")[0] == prefix for p in pids])
        test_idx   = all_idx[test_mask]
        train_idx  = all_idx[~test_mask]

        # ── Validation ─────────────────────────────────────────────────────
        test_pfx_set  = set(p.split(":")[0] for p in pids[test_idx])
        train_pfx_set = set(p.split(":")[0] for p in pids[train_idx])

        assert test_pfx_set == {prefix}, \
            f"{ds_name}: test contains other prefixes: {test_pfx_set}"
        assert prefix not in train_pfx_set, \
            f"{ds_name}: train contains held-out prefix '{prefix}'"
        assert len(set(train_idx) & set(test_idx)) == 0, \
            f"{ds_name}: train/test overlap"
        assert len(train_idx) + len(test_idx) == n_total, \
            f"{ds_name}: train+test ({len(train_idx)+len(test_idx)}) != n_total ({n_total})"
        assert test_idx.min() >= 0 and test_idx.max() < n_total, \
            f"{ds_name}: test_idx out of range"
        assert train_idx.min() >= 0 and train_idx.max() < n_total, \
            f"{ds_name}: train_idx out of range"

        # Confirm train side covers all 28 other datasets
        expected_train_prefixes = known_prefixes - {prefix}
        # Filter to prefixes that actually exist in cache
        expected_train_prefixes &= actual_prefixes
        missing = expected_train_prefixes - train_pfx_set
        if missing:
            failed.append((ds_name, f"train missing datasets: {missing}"))
            continue

        # ── Save ──────────────────────────────────────────────────────────
        dest_fold = DEST_DIR / f"fold_{ds_name}"
        dest_fold.mkdir(parents=True, exist_ok=True)
        np.savez(dest_fold / "split_indices.npz",
                 train=train_idx.astype(np.int64),
                 test=test_idx.astype(np.int64),
                 seed=42,
                 lodo_held_out=ds_name)

        print(f"  OK   {ds_name:<35s} train={len(train_idx):>6}  test={len(test_idx):>5}  "
              f"prefix='{prefix}'")
        written += 1

    print(f"\nWrote {written}/{len(DATASETS_ALL)} folds.")
    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
