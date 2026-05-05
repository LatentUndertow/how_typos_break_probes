#!/usr/bin/env python3
"""
Combined perturbation evaluation for 5-fold CV probes.

Single Llama load, one pass over all samples. Each sample belongs to exactly
one fold's test set. Scores clean + perturbed with that fold's probes.

Outputs:
  {cv_dir}/perteval/all_scores.npz     — all folds concatenated
  {cv_dir}/perteval/fold_{i}_scores.npz — per-fold scores
  {cv_dir}/perteval/summary.json       — metrics mean ± std across folds

Usage:
    python perteval_5fold.py --cv-dir results/5fold_cv
    python perteval_5fold.py --cv-dir results/5fold_cv --conditions full_bundle every_second_word
    python perteval_5fold.py --cv-dir results/5fold_cv --n-test 500
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

from activation_robustness.experiments.perturbation_eval_probes import PERTURBATION_FNS, DATASETS, load_probe

CACHE_DIR = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_metrics(scores, labels, thr=0.5):
    from sklearn.metrics import roc_auc_score, accuracy_score
    valid = ~np.isnan(scores)
    s, l = scores[valid], labels[valid]
    if len(np.unique(l)) < 2:
        preds = (s >= thr).astype(int)
        if l[0] == 0:
            fpr = preds.mean()
            return {"auc": float("nan"), "acc": float(1 - fpr), "tpr": float("nan"), "fpr": float(fpr), "n": int(valid.sum())}
        else:
            tpr = preds.mean()
            return {"auc": float("nan"), "acc": float(tpr), "tpr": float(tpr), "fpr": float("nan"), "n": int(valid.sum())}
    preds = (s >= thr).astype(int)
    tp = int(((preds==1)&(l==1)).sum()); fn = int(((preds==0)&(l==1)).sum())
    fp = int(((preds==1)&(l==0)).sum()); tn = int(((preds==0)&(l==0)).sum())
    return {
        "auc": float(roc_auc_score(l, s)),
        "acc": float(accuracy_score(l, preds)),
        "tpr": tp / max(tp+fn, 1),
        "fpr": fp / max(fp+tn, 1),
        "n":   int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir",     required=True,
                        help="Path to 5fold_cv root dir containing fold_0..fold_4")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: cv_dir/perteval)")
    parser.add_argument("--conditions", nargs="+",
                        default=["full_bundle", "every_second_word"])
    parser.add_argument("--probe-types", nargs="+", default=None,
                        help="Restrict to specific probe types (default: all in fold dirs)")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Restrict to specific fold names (e.g. fold_EnronDataset). "
                             "Useful for sharding across GPUs.")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Limit test samples per fold (for quick testing)")
    args = parser.parse_args()

    cv_dir    = Path(args.cv_dir)
    out_dir   = Path(args.output_dir) if args.output_dir else cv_dir / "perteval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng    = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover folds
    fold_dirs = sorted([
        d for d in cv_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_")
        and (d / "split_indices.npz").exists()
    ])
    if args.folds:
        # Match by exact dir name OR by suffix after "fold_"
        wanted = set(args.folds)
        fold_dirs = [d for d in fold_dirs
                     if d.name in wanted or d.name.split("_", 1)[-1] in wanted]
        log(f"Filtered to {len(fold_dirs)} folds: {[d.name for d in fold_dirs]}")
    log(f"Found {len(fold_dirs)} trained folds in {cv_dir}")
    if not fold_dirs:
        log("No folds found. Run run_5fold_sweep.sh first.")
        return

    # Load model
    log("Loading model...")
    extractor = ActivationExtractor(
        model_name=MODEL_NAME, layer=LAYER,
        max_seq_len=16384, attn_implementation="sdpa",
    )
    log("Model loaded")

    # Load all texts
    log("Loading dataset texts...")
    loader    = DataLoader(tokenizer=extractor.tokenizer, add_generation_prompt=True)
    all_texts = {}
    for entry in DATASETS:
        ds_name = entry["class"] if isinstance(entry, dict) else entry
        try:
            samples = loader.load(entry, verbose=False)
            for s in samples:
                all_texts[s["prompt_id"]] = s["text"]
            log(f"  {ds_name}: {len(samples)} texts")
        except Exception as e:
            log(f"  {ds_name}: SKIP ({e})")

    # Load full cache (fixed order) for prompt_id and label lookup
    cache   = ActivationCache(cache_dir=CACHE_DIR)
    full_ds = CachedActivationDataset(cache, DATASETS_ALL)
    log(f"Full dataset: {len(full_ds)} samples")

    # Load probes and test indices per fold
    log("Loading fold probes...")
    fold_probes = {}   # fold_name -> {probe_name: clf}
    fold_meta   = {}   # fold_name -> {test_idx, labels}

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        split     = np.load(fold_dir / "split_indices.npz")
        test_idx  = split["test"]
        if args.n_test:
            test_idx = test_idx[:args.n_test]

        probes = {}
        for d in sorted(fold_dir.iterdir()):
            if d.is_dir() and (d / "probe.pt").exists():
                if args.probe_types and d.name not in args.probe_types:
                    continue
                clf = load_probe(d, extractor, device)
                if clf is not None:
                    probes[d.name] = clf
        if not probes:
            log(f"  {fold_name}: no probes, skipping")
            continue

        fold_probes[fold_name] = probes
        fold_meta[fold_name]   = {
            "test_idx": test_idx,
            "labels":   full_ds.labels[test_idx].numpy(),
        }
        log(f"  {fold_name}: {len(probes)} probes, {len(test_idx)} test samples")

    if not fold_probes:
        log("No fold probes loaded. Aborting.")
        return

    probe_names = sorted({p for probes in fold_probes.values() for p in probes})
    conditions  = {c: PERTURBATION_FNS[c] for c in args.conditions if c in PERTURBATION_FNS}
    log(f"Probes: {probe_names}")
    log(f"Conditions: {list(conditions.keys())}")

    # Build global sample → fold mapping (non-overlapping test sets)
    sample_to_fold = {}
    for fn, meta in fold_meta.items():
        for idx in meta["test_idx"]:
            sample_to_fold[int(idx)] = fn

    all_global_idx = sorted(sample_to_fold.keys())
    log(f"\nTotal samples to score: {len(all_global_idx)}")

    pid_map = {int(i): full_ds.prompt_ids[i] for i in all_global_idx}

    # Score arrays per fold × probe × condition
    clean_scores = {
        fn: {p: np.full(len(fold_meta[fn]["test_idx"]), np.nan) for p in probe_names}
        for fn in fold_probes
    }
    pert_scores = {
        fn: {c: {p: np.full(len(fold_meta[fn]["test_idx"]), np.nan) for p in probe_names}
             for c in conditions}
        for fn in fold_probes
    }
    fold_local_idx = {
        fn: {int(gidx): li for li, gidx in enumerate(fold_meta[fn]["test_idx"])}
        for fn in fold_meta
    }

    # Resume from existing per-fold npz files if present.
    # Loads any previously-computed scores back into the arrays so the main
    # loop skips samples that are already filled in for every probe×condition.
    n_resumed = 0
    for fn in fold_probes:
        npz_path = out_dir / f"{fn}_scores.npz"
        if not npz_path.exists():
            continue
        existing = np.load(npz_path)
        for pname in probe_names:
            ck = f"{pname}_clean"
            if ck in existing.files and existing[ck].shape == clean_scores[fn][pname].shape:
                clean_scores[fn][pname] = existing[ck].astype(np.float64)
            for cname in conditions:
                pk = f"{pname}_{cname}"
                if pk in existing.files and existing[pk].shape == pert_scores[fn][cname][pname].shape:
                    pert_scores[fn][cname][pname] = existing[pk].astype(np.float64)
        # Count how many samples in this fold are fully filled.
        fold_done = np.ones(len(fold_meta[fn]["test_idx"]), dtype=bool)
        for pname in probe_names:
            fold_done &= np.isfinite(clean_scores[fn][pname])
            for cname in conditions:
                fold_done &= np.isfinite(pert_scores[fn][cname][pname])
        n_resumed += int(fold_done.sum())
    if n_resumed > 0:
        log(f"Resumed: {n_resumed} samples already fully scored, will skip them")

    def _save_partial():
        """Persist current score arrays. Safe to call mid-run for resumability."""
        for fn, meta in fold_meta.items():
            if fn not in fold_probes:
                continue
            save = {"labels": meta["labels"]}
            for p in probe_names:
                save[f"{p}_clean"] = clean_scores[fn][p]
                for c in conditions:
                    save[f"{p}_{c}"] = pert_scores[fn][c][p]
            np.savez_compressed(out_dir / f"{fn}_scores.npz", **save)

    # Single pass
    log(f"\nScoring {len(all_global_idx)} samples...")
    t0       = time.time()
    n_done   = 0
    n_skipped = 0
    SAVE_EVERY = 5000

    for global_idx in all_global_idx:
        fn      = sample_to_fold[global_idx]
        local_i = fold_local_idx[fn][global_idx]
        pid     = pid_map[global_idx]

        if pid not in all_texts:
            n_done += 1
            continue

        # Resume: skip if every score for this sample is already filled in.
        already_done = all(
            np.isfinite(clean_scores[fn][p][local_i]) for p in probe_names
        ) and all(
            np.isfinite(pert_scores[fn][c][p][local_i])
            for c in conditions for p in probe_names
        )
        if already_done:
            n_done += 1
            n_skipped += 1
            continue

        clean_text = all_texts[pid]
        probes     = fold_probes[fn]

        pert_texts = {}
        for cname, pfn in conditions.items():
            pt = pfn(clean_text, rng)
            if pt is not None:
                pert_texts[cname] = pt

        # Load clean activations from cache (no LLM forward pass needed).
        # Cache stores per-token activations from the same model+layer used here,
        # so this is identical to extract_all_positions([clean_text]) but ~16x faster.
        with torch.no_grad():
            hidden_cached, _, _, _ = full_ds.load_batch([global_idx], max_tokens=None)
            h = hidden_cached.float().to(device)
            for pname in probe_names:
                clf = probes.get(pname)
                if clf is None:
                    continue
                x = h[:, clf._eval_slice[0]:clf._eval_slice[1], :] if clf._eval_slice else h
                if clf._eval_mean is not None:
                    x = (x - clf._eval_mean) / clf._eval_std
                clean_scores[fn][pname][local_i] = torch.sigmoid(clf._model(x)).cpu().item()
            del hidden_cached, h

        for cname, pt in pert_texts.items():
            with torch.no_grad():
                hidden, _ = extractor.extract_all_positions([pt])
                h = hidden.float()
                for pname in probe_names:
                    clf = probes.get(pname)
                    if clf is None:
                        continue
                    x = h[:, clf._eval_slice[0]:clf._eval_slice[1], :] if clf._eval_slice else h
                    if clf._eval_mean is not None:
                        x = (x - clf._eval_mean) / clf._eval_std
                    pert_scores[fn][cname][pname][local_i] = torch.sigmoid(clf._model(x)).cpu().item()
                del hidden, h

        n_done += 1
        if n_done % 500 == 0:
            elapsed = time.time() - t0
            rate    = n_done / elapsed
            eta     = (len(all_global_idx) - n_done) / rate / 60
            log(f"  [{n_done}/{len(all_global_idx)}] {rate:.1f} samples/sec, ETA {eta:.0f} min "
                f"(skipped {n_skipped} resumed)")
        if n_done % SAVE_EVERY == 0:
            _save_partial()
            log(f"  checkpoint saved at sample {n_done}")

    log(f"Scoring done in {time.time()-t0:.0f}s")

    # Save per-fold npz files
    for fn, meta in fold_meta.items():
        if fn not in fold_probes:
            continue
        save = {"labels": meta["labels"]}
        for p in probe_names:
            save[f"{p}_clean"] = clean_scores[fn][p]
            for c in conditions:
                save[f"{p}_{c}"] = pert_scores[fn][c][p]
        np.savez_compressed(out_dir / f"{fn}_scores.npz", **save)

    # Aggregate metrics across folds: mean ± std per probe per condition
    summary = {}
    for p in probe_names:
        summary[p] = {"clean": {}, "conditions": {}}
        clean_mets = []
        for fn, meta in fold_meta.items():
            if fn not in fold_probes or p not in fold_probes[fn]:
                continue
            m = compute_metrics(clean_scores[fn][p], meta["labels"])
            clean_mets.append(m)

        for metric in ["auc", "acc", "tpr", "fpr"]:
            vals = [m[metric] for m in clean_mets if not (m[metric] != m[metric])]  # drop nan
            summary[p]["clean"][metric] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std":  float(np.std(vals))  if vals else float("nan"),
                "per_fold": vals,
            }

        for cname in conditions:
            summary[p]["conditions"][cname] = {}
            delta_tpr, delta_fpr = [], []
            for fn, meta in fold_meta.items():
                if fn not in fold_probes or p not in fold_probes[fn]:
                    continue
                valid = ~np.isnan(clean_scores[fn][p]) & ~np.isnan(pert_scores[fn][cname][p])
                sc = clean_scores[fn][p][valid]
                sp = pert_scores[fn][cname][p][valid]
                l  = meta["labels"][valid]
                mc = compute_metrics(sc, l)
                mp = compute_metrics(sp, l)
                if not (mc["tpr"] != mc["tpr"]) and not (mp["tpr"] != mp["tpr"]):
                    delta_tpr.append(mp["tpr"] - mc["tpr"])
                if not (mc["fpr"] != mc["fpr"]) and not (mp["fpr"] != mp["fpr"]):
                    delta_fpr.append(mp["fpr"] - mc["fpr"])

            summary[p]["conditions"][cname]["delta_tpr"] = {
                "mean": float(np.mean(delta_tpr)) if delta_tpr else float("nan"),
                "std":  float(np.std(delta_tpr))  if delta_tpr else float("nan"),
                "per_fold": delta_tpr,
            }
            summary[p]["conditions"][cname]["delta_fpr"] = {
                "mean": float(np.mean(delta_fpr)) if delta_fpr else float("nan"),
                "std":  float(np.std(delta_fpr))  if delta_fpr else float("nan"),
                "per_fold": delta_fpr,
            }

    # Print summary table
    log("\n" + "="*70)
    log("SUMMARY  (mean ± std across folds)")
    log("="*70)
    log(f"{'Probe':32s}  {'AUC':12s}  {'TPR':12s}  {'FPR':12s}")
    log("-"*70)
    for p, m in summary.items():
        c = m["clean"]
        auc = f"{c['auc']['mean']:.4f}±{c['auc']['std']:.4f}"
        tpr = f"{c['tpr']['mean']:.4f}±{c['tpr']['std']:.4f}"
        fpr = f"{c['fpr']['mean']:.4f}±{c['fpr']['std']:.4f}"
        log(f"{p:32s}  {auc:12s}  {tpr:12s}  {fpr:12s}")
        for cname, cd in m["conditions"].items():
            dt = cd["delta_tpr"]
            df = cd["delta_fpr"]
            log(f"  [{cname}] ΔTPR={dt['mean']:+.4f}±{dt['std']:.4f}  ΔFPR={df['mean']:+.4f}±{df['std']:.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    log(f"\nSummary saved to {out_dir / 'summary.json'}")

    # Also save concatenated all_scores.npz for downstream use
    all_labels_cat = np.concatenate([fold_meta[fn]["labels"] for fn in fold_meta if fn in fold_probes])
    all_save = {"labels": all_labels_cat}
    for p in probe_names:
        all_save[f"{p}_clean"] = np.concatenate([
            clean_scores[fn][p] for fn in fold_meta if fn in fold_probes
        ])
        for c in conditions:
            all_save[f"{p}_{c}"] = np.concatenate([
                pert_scores[fn][c][p] for fn in fold_meta if fn in fold_probes
            ])
    np.savez_compressed(out_dir / "all_scores.npz", **all_save)
    log(f"All scores saved to {out_dir / 'all_scores.npz'}")
    log("Done.")


if __name__ == "__main__":
    main()
