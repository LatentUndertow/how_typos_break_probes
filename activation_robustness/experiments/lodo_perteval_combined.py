#!/usr/bin/env python3
"""
Combined LODO perturbation evaluation.

Instead of loading Llama once per fold (29x), this script does a single pass
over all 168K samples. Each sample belongs to exactly one LODO fold (the fold
that held it out as test). We extract activations once per sample (clean +
perturbed), then score with that fold's probe set.

Total cost: 1 Llama load + 168K × (1 + N_conditions) extractions
vs naive:   29 Llama loads + Σ fold_test_size × (1 + N_conditions) extractions

Usage:
    python lodo_perteval_combined.py
    python lodo_perteval_combined.py --conditions full_bundle every_second_word
    python lodo_perteval_combined.py --n-test 500  # quick sanity check
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from activation_robustness.data.activation_extractor import ActivationExtractor
from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.data.data_loader import DataLoader
from activation_robustness.probes.architectures import MultiArchProbe, ProbeConfig

CACHE_DIR = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
LODO_DIR  = _REPO / "activation_robustness" / "results" / "lodo_sweep"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 31
SEED  = 42

DATASETS_ALL = [
    "EnronDataset", "Dolly15kDataset", "OpenOrcaDataset", "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    "BIPIADataset", "InjecAgentDataset", "LLMailDataset", "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset",
    "HarmBenchDataset", "AgentDojoDataset", "APIGenMTDataset",
    "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset",
    "ScamDataset", "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

# Same params as perturbation_eval_probes.py for text loading
DATASETS_SPEC = [
    {"class": "EnronDataset",    "params": {"include_email_format": True}},
    "Dolly15kDataset",
    {"class": "OpenOrcaDataset", "params": {"include_system_prompt": True}},
    "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    {"class": "BIPIADataset",       "params": {"bipia_root": "./data/BIPIA",
                                               "task_names": ["email", "code", "table"]}},
    {"class": "InjecAgentDataset",  "params": {"injecagent_root": "./data/InjecAgent",
                                               "attack_types": ["dh", "ds"], "setting": "base"}},
    {"class": "LLMailDataset",   "params": {"include_email_format": True}},
    "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset",
    "HarmBenchDataset", "AgentDojoDataset", "APIGenMTDataset",
    "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset",
    {"class": "ScamDataset", "params": {"scam_root": "./data/SCAM"}},
    "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

import re
from activation_robustness.experiments.perturbation_eval_probes import (
    PERTURBATION_FNS, load_probe, load_test_texts, log,
)


def load_fold_probes(fold_dir: Path, extractor, device):
    """Load all probe.pt files from a fold directory."""
    probes = {}
    for d in sorted(fold_dir.iterdir()):
        if not d.is_dir() or not (d / "probe.pt").exists():
            continue
        clf = load_probe(d, extractor, device)
        if clf is not None:
            probes[d.name] = clf
    return probes


def compute_metrics(scores, labels, thr=0.5):
    valid = ~np.isnan(scores)
    s, l = scores[valid], labels[valid]
    if len(np.unique(l)) < 2:
        preds = (s >= thr).astype(int)
        if l[0] == 0:  # all benign
            fp = int(preds.sum()); tn = int((~preds.astype(bool)).sum())
            fpr = fp / max(fp + tn, 1)
            return {"auc": float("nan"), "acc": float(1 - fpr), "tpr": float("nan"), "fpr": fpr, "n": int(valid.sum())}
        else:  # all malicious
            tp = int(preds.sum()); fn = int((~preds.astype(bool)).sum())
            tpr = tp / max(tp + fn, 1)
            return {"auc": float("nan"), "acc": tpr, "tpr": tpr, "fpr": float("nan"), "n": int(valid.sum())}
    from sklearn.metrics import roc_auc_score, accuracy_score
    preds = (s >= thr).astype(int)
    tp = int(((preds==1)&(l==1)).sum()); fn = int(((preds==0)&(l==1)).sum())
    fp = int(((preds==1)&(l==0)).sum()); tn = int(((preds==0)&(l==0)).sum())
    return {
        "auc":  float(roc_auc_score(l, s)),
        "acc":  float(accuracy_score(l, preds)),
        "tpr":  tp / max(tp+fn, 1),
        "fpr":  fp / max(fp+tn, 1),
        "n":    int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["full_bundle", "every_second_word"])
    parser.add_argument("--n-test", type=int, default=None,
                        help="Limit samples per fold for quick testing")
    parser.add_argument("--lodo-dir", type=str, default=None)
    args = parser.parse_args()

    lodo_dir = Path(args.lodo_dir) if args.lodo_dir else LODO_DIR
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover completed folds
    fold_dirs = sorted([
        d for d in lodo_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_") and (d / "split_indices.npz").exists()
    ])
    log(f"Found {len(fold_dirs)} completed folds")

    if not fold_dirs:
        log("No completed folds found. Run LODO training first.")
        return

    # Load model once
    log("Loading model...")
    extractor = ActivationExtractor(
        model_name=MODEL_NAME, layer=LAYER,
        max_seq_len=16384, attn_implementation="sdpa",
    )
    log("Model loaded")

    # Load all texts once
    log("Loading dataset texts...")
    loader = DataLoader(tokenizer=extractor.tokenizer, add_generation_prompt=True)
    all_texts = {}
    for entry in DATASETS_SPEC:
        ds_name = entry["class"] if isinstance(entry, dict) else entry
        try:
            samples = loader.load(entry, verbose=False)
            for s in samples:
                all_texts[s["prompt_id"]] = s["text"]
            log(f"  {ds_name}: {len(samples)} texts")
        except Exception as e:
            log(f"  {ds_name}: SKIP ({e})")

    # Load full cache for prompt_id and dataset_index lookup
    cache = ActivationCache(cache_dir=CACHE_DIR)
    full_ds = CachedActivationDataset(cache, DATASETS_ALL)
    log(f"Full dataset: {len(full_ds)} samples")

    # Build dataset_name -> index in DATASETS_ALL
    ds_name_to_pos = {name: i for i, name in enumerate(DATASETS_ALL)}

    # Build dataset_name -> global indices in full_ds (fixed order)
    # Each fold's test set = all samples from the held-out dataset in full_ds.
    # We derive this from full_ds._dataset_indices rather than split_indices.npz,
    # because the sweep reorders datasets (held-out last) before saving the split,
    # making the saved test_idx invalid against our fixed-order full_ds.
    ds_idx_arr = full_ds._dataset_indices.numpy()

    # Load probes for each fold
    log("Loading fold probes...")
    fold_probes = {}   # fold_name -> {probe_name: clf}
    fold_meta   = {}   # fold_name -> {held_out, test_idx, labels}
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name  # e.g. fold_EnronDataset
        held_out  = fold_name[len("fold_"):]

        if held_out not in ds_name_to_pos:
            log(f"  {fold_name}: held-out dataset not in full_ds, skipping")
            continue

        # Global indices of held-out dataset samples in fixed-order full_ds
        ds_pos   = ds_name_to_pos[held_out]
        test_idx = np.where(ds_idx_arr == ds_pos)[0]
        if args.n_test:
            test_idx = test_idx[:args.n_test]

        probes = load_fold_probes(fold_dir, extractor, device)
        if not probes:
            log(f"  {fold_name}: no probes found, skipping")
            continue

        fold_probes[fold_name] = probes
        fold_meta[fold_name]   = {
            "held_out": held_out,
            "test_idx": test_idx,
            "labels":   full_ds.labels[test_idx].numpy(),
        }
        log(f"  {fold_name}: {len(probes)} probes, {len(test_idx)} test samples")

    if not fold_probes:
        log("No fold probes loaded.")
        return

    # Build global sample → fold mapping
    # Each sample in a fold's test set belongs exclusively to that fold
    sample_to_fold = {}  # global_idx -> fold_name
    for fold_name, meta in fold_meta.items():
        for idx in meta["test_idx"]:
            sample_to_fold[int(idx)] = fold_name

    all_global_idx = sorted(sample_to_fold.keys())
    log(f"\nTotal samples to score: {len(all_global_idx)}")

    # Map global indices to prompt_ids for text lookup
    pid_map = {int(i): full_ds.prompt_ids[i] for i in all_global_idx}

    # Validate conditions
    conditions = {c: PERTURBATION_FNS[c] for c in args.conditions if c in PERTURBATION_FNS}
    log(f"Conditions: {list(conditions.keys())}")

    # Initialize score arrays per fold per probe per condition
    probe_names_per_fold = {fn: list(fp.keys()) for fn, fp in fold_probes.items()}
    clean_scores = {
        fn: {pn: np.full(len(fold_meta[fn]["test_idx"]), np.nan)
             for pn in probe_names_per_fold[fn]}
        for fn in fold_probes
    }
    pert_scores = {
        fn: {cond: {pn: np.full(len(fold_meta[fn]["test_idx"]), np.nan)
                    for pn in probe_names_per_fold[fn]}
             for cond in conditions}
        for fn in fold_probes
    }
    # local index within fold's test_idx
    fold_local_idx = {
        fn: {int(gidx): li for li, gidx in enumerate(fold_meta[fn]["test_idx"])}
        for fn in fold_meta
    }

    # Single pass over all samples
    log(f"\nScoring {len(all_global_idx)} samples...")
    t0 = time.time()
    n_done = 0

    for global_idx in all_global_idx:
        fold_name = sample_to_fold[global_idx]
        local_i   = fold_local_idx[fold_name][global_idx]
        pid       = pid_map[global_idx]

        if pid not in all_texts:
            n_done += 1
            continue

        clean_text = all_texts[pid]
        probes     = fold_probes[fold_name]

        # Pre-apply perturbations (text level)
        pert_texts = {}
        for cond_name, perturb_fn in conditions.items():
            pert = perturb_fn(clean_text, rng)
            if pert is not None:
                pert_texts[cond_name] = pert

        # Extract clean + score all probes
        with torch.no_grad():
            hidden, _ = extractor.extract_all_positions([clean_text])
            h = hidden.float()
            for pname, clf in probes.items():
                x = h[:, clf._eval_slice[0]:clf._eval_slice[1], :] if clf._eval_slice else h
                if clf._eval_mean is not None:
                    x = (x - clf._eval_mean) / clf._eval_std
                score = torch.sigmoid(clf._model(x)).cpu().item()
                clean_scores[fold_name][pname][local_i] = score
            del hidden, h

        # Extract perturbed + score
        for cond_name, pert_text in pert_texts.items():
            with torch.no_grad():
                hidden, _ = extractor.extract_all_positions([pert_text])
                h = hidden.float()
                for pname, clf in probes.items():
                    x = h[:, clf._eval_slice[0]:clf._eval_slice[1], :] if clf._eval_slice else h
                    if clf._eval_mean is not None:
                        x = (x - clf._eval_mean) / clf._eval_std
                    score = torch.sigmoid(clf._model(x)).cpu().item()
                    pert_scores[fold_name][cond_name][pname][local_i] = score
                del hidden, h

        n_done += 1
        if n_done % 500 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta  = (len(all_global_idx) - n_done) / rate / 60
            log(f"  [{n_done}/{len(all_global_idx)}] {rate:.1f} samples/sec, ETA {eta:.0f} min")

    log(f"Scoring done in {time.time()-t0:.0f}s")

    # Compute and report metrics per fold per condition per probe
    summary = {}
    for fold_name, meta in fold_meta.items():
        if fold_name not in fold_probes:
            continue
        held_out = meta["held_out"]
        labels   = meta["labels"]
        summary[held_out] = {}

        log(f"\n{'='*60}")
        log(f"  Fold: {held_out}")
        log(f"{'='*60}")

        for pname in probe_names_per_fold[fold_name]:
            c_scores = clean_scores[fold_name][pname]
            c_met = compute_metrics(c_scores, labels)
            summary[held_out][pname] = {"clean": c_met, "conditions": {}}

            log(f"  {pname}:")
            log(f"    Clean: acc={c_met['acc']:.4f} auc={c_met['auc']:.4f} "
                f"tpr={c_met['tpr']:.4f} fpr={c_met['fpr']:.4f} n={c_met['n']}")

            for cond_name in conditions:
                p_scores = pert_scores[fold_name][cond_name][pname]
                # align valid mask
                valid = ~np.isnan(c_scores) & ~np.isnan(p_scores)
                p_met = compute_metrics(p_scores[valid], labels[valid])
                c_met_v = compute_metrics(c_scores[valid], labels[valid])
                summary[held_out][pname]["conditions"][cond_name] = {
                    "clean": c_met_v, "pert": p_met,
                    "delta_auc": (p_met["auc"] - c_met_v["auc"])
                                  if not (np.isnan(p_met["auc"]) or np.isnan(c_met_v["auc"])) else float("nan"),
                    "delta_tpr": p_met["tpr"] - c_met_v["tpr"]
                                  if not (np.isnan(p_met["tpr"]) or np.isnan(c_met_v["tpr"])) else float("nan"),
                    "delta_fpr": p_met["fpr"] - c_met_v["fpr"]
                                  if not (np.isnan(p_met["fpr"]) or np.isnan(c_met_v["fpr"])) else float("nan"),
                }
                log(f"    {cond_name}: pert_auc={p_met['auc']:.4f} "
                    f"ΔTPR={p_met['tpr']-c_met_v['tpr']:+.4f} "
                    f"ΔFPR={p_met['fpr']-c_met_v['fpr']:+.4f}")

    # Save per-fold npz + summary json
    for fold_name, meta in fold_meta.items():
        if fold_name not in fold_probes:
            continue
        fold_dir  = lodo_dir / fold_name
        out_dir   = fold_dir / "perturbation_eval"
        out_dir.mkdir(exist_ok=True)

        save = {"labels": meta["labels"]}
        for pname in probe_names_per_fold[fold_name]:
            save[f"{pname}_clean"] = clean_scores[fold_name][pname]
            for cond_name in conditions:
                save[f"{pname}_{cond_name}"] = pert_scores[fold_name][cond_name][pname]
        np.savez_compressed(out_dir / "all_scores.npz", **save)

    with open(lodo_dir / "lodo_perteval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {lodo_dir / 'lodo_perteval_summary.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
