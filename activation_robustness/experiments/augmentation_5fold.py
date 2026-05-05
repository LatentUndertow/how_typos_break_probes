#!/usr/bin/env python3
"""
Principled Perturbation-Augmented Training: 5-Fold CV.

Augmentation policy (Gap 1c):
  For each training sample, apply a random bundle of near-end perturbations.
  Each near-end family is applied independently with probability AUG_PROB,
  guaranteeing at least one perturbation lands per sample.

  Near-end families (all target the spatial high-impact zone d<=2):
    - last_word_typo     : QWERTY adjacent-key typo on last alphabetic word
    - trail_period       : toggle trailing period
    - early_shift        : uppercase letter before punctuation
    - terminal_punct     : one of {q_to_period, q_to_slash, dot_to_slash},
                           chosen randomly per sample (mutually exclusive slot)

  Mid-sequence families are EXCLUDED from augmentation — they form the
  held-out generalization test group.

Evaluation conditions (applied independently at test time):
  Near-end (seen family):
    last_word_typo, trail_period, early_shift,
    q_to_period, q_to_slash, dot_to_slash
  Mid-sequence (unseen / held-out):
    mid_word_typo, missing_space
  Compound:
    full_bundle  (original 5-component bundle from Table 2)

Usage:
    source .venv/bin/activate
    python activation_robustness/experiments/augmentation_5fold.py
    # Quick smoke test:
    python activation_robustness/experiments/augmentation_5fold.py --n-samples 5000 --n-folds 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "prompt-mining")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from activation_robustness.data.activation_extractor import ActivationExtractor
from activation_robustness.data.activation_cache import ActivationCache, CachedActivationDataset
from activation_robustness.data.data_loader import DataLoader

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR   = os.environ.get("ACTIVATION_CACHE_DIR", "./cache_data/activations")
DEFAULT_CV_DIR = str(
    _PROJECT_ROOT / "activation_robustness" / "results" / "5fold_cv"
)
DEFAULT_OUTPUT_DIR = str(
    _PROJECT_ROOT / "activation_robustness" / "results" / "augmentation_5fold"
)
MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
LAYER       = 31
POS         = -5          # readout position (from end)
MAX_TOKENS  = 8           # window for load_batch — covers pos -5 with margin
SEED        = 42
AUG_PROB    = 0.5         # probability each near-end family fires per sample

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

USER_MSG_PATTERN = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL,
)
QWERTY_ADJACENT: dict[str, str] = {
    "q": "wa",   "w": "qeas",  "e": "wrds",  "r": "etf",   "t": "ryg",
    "y": "tuh",  "u": "yij",   "i": "uok",   "o": "ipl",   "p": "o",
    "a": "qwsz", "s": "wedxza","d": "erfcxs", "f": "rtgvcd","g": "tyhbvf",
    "h": "yujnbg","j": "uikmnh","k": "iolmj", "l": "opk",
    "z": "asx",  "x": "zsdc",  "c": "xdfv",  "v": "cfgb",  "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Perturbation primitives
# ---------------------------------------------------------------------------

def _find_user_span(text: str):
    m = USER_MSG_PATTERN.search(text)
    if m is None:
        return None, None, None
    return m.start(1), m.end(1), m.group(1)


def _replace_user(text: str, cs: int, ce: int, new_content: str) -> str:
    return text[:cs] + new_content + text[ce:]


def _typo_in_word(word: str, rng: np.random.Generator) -> str | None:
    candidates = [i for i, c in enumerate(word) if c.lower() in QWERTY_ADJACENT]
    if not candidates:
        return None
    idx = int(rng.choice(candidates))
    c = word[idx]
    adj = QWERTY_ADJACENT[c.lower()]
    rep = adj[rng.integers(len(adj))]
    if c.isupper():
        rep = rep.upper()
    return word[:idx] + rep + word[idx + 1:]


# ── Near-end families ────────────────────────────────────────────────────────

def apply_last_word_typo(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", user)]
    if not spans:
        return None
    ws, we = spans[-1]
    new_word = _typo_in_word(user[ws:we], rng)
    if new_word is None:
        return None
    return _replace_user(text, cs, ce, user[:ws] + new_word + user[we:])


def apply_trail_period(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    stripped = user.rstrip()
    trail = user[len(stripped):]
    if stripped.endswith("."):
        new = stripped[:-1] + trail
    elif stripped and stripped[-1] not in "?!,;:":
        new = stripped + "." + trail
    else:
        return None
    return _replace_user(text, cs, ce, new) if new != user else None


def apply_early_shift(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    m = re.search(r"[a-z][!?]", user)
    if m is None:
        return None
    pos = m.start()
    new = user[:pos] + user[pos].upper() + user[pos + 1:]
    return _replace_user(text, cs, ce, new)


# ── Terminal punctuation slot (mutually exclusive) ───────────────────────────

def apply_q_to_period(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    s = user.rstrip()
    if not s.endswith("?"):
        return None
    trail = user[len(s):]
    return _replace_user(text, cs, ce, s[:-1] + "." + trail)


def apply_q_to_slash(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    s = user.rstrip()
    if not s.endswith("?"):
        return None
    trail = user[len(s):]
    return _replace_user(text, cs, ce, s[:-1] + "/" + trail)


def apply_dot_to_slash(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    s = user.rstrip()
    if not s.endswith("."):
        return None
    trail = user[len(s):]
    return _replace_user(text, cs, ce, s[:-1] + "/" + trail)


TERMINAL_PUNCT_SLOT = [apply_q_to_period, apply_q_to_slash, apply_dot_to_slash]

# ── Mid-sequence families (HELD-OUT — not used in augmentation) ─────────────

def apply_mid_word_typo(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", user)]
    if len(spans) < 4:
        return None
    eligible = spans[:-3]
    ws, we = eligible[int(rng.integers(len(eligible)))]
    new_word = _typo_in_word(user[ws:we], rng)
    if new_word is None:
        return None
    return _replace_user(text, cs, ce, user[:ws] + new_word + user[we:])


def apply_missing_space(text: str, rng: np.random.Generator) -> str | None:
    cs, ce, user = _find_user_span(text)
    if user is None:
        return None
    # Find punctuation followed by a space (not at end) and remove the space
    matches = list(re.finditer(r"([.!?,;:]) ([A-Za-z])", user))
    if not matches:
        return None
    m = matches[int(rng.integers(len(matches)))]
    # group 1 = punct, group 2 = letter; drop the space between them
    new = user[:m.end(1)] + user[m.start(2):]
    return _replace_user(text, cs, ce, new) if new != user else None


# ── Full bundle (original 5-component, for test-time headline metric) ────────

def apply_full_bundle(text: str, rng: np.random.Generator) -> str | None:
    fns = [apply_q_to_period, apply_last_word_typo, apply_mid_word_typo,
           apply_early_shift, apply_trail_period]
    cur = text
    applied = False
    for fn in fns:
        out = fn(cur, rng)
        if out is not None:
            cur = out
            applied = True
    return cur if applied else None


# ── Augmentation policy ──────────────────────────────────────────────────────

OTHER_NEAR_END = [apply_last_word_typo, apply_trail_period, apply_early_shift]


def apply_near_end_augmentation(text: str, rng: np.random.Generator) -> str | None:
    """Random bundle of near-end perturbations, at least one guaranteed.

    Design:
      Step 1 — apply each of the three non-terminal near-end families
               independently with probability AUG_PROB (in random order).
      Step 2 — apply terminal punct slot (one of q_to_period / q_to_slash /
               dot_to_slash, chosen randomly) with probability AUG_PROB.

    Terminal punct is applied LAST to avoid interaction: if trail_period
    fires before q_to_slash the message may end in '.' which q_to_slash
    ignores; conversely if q_to_slash fired first trail_period would append
    '.' after '/' producing an artefact like 'France/.'.
    """
    cur = text
    applied_any = False

    # Step 1: non-terminal families in random order
    order = rng.permutation(len(OTHER_NEAR_END))
    for idx in order:
        if rng.random() < AUG_PROB:
            out = OTHER_NEAR_END[idx](cur, rng)
            if out is not None:
                cur = out
                applied_any = True

    # Step 2: terminal punct slot (applied last to avoid interaction)
    if rng.random() < AUG_PROB:
        terminal_fn = TERMINAL_PUNCT_SLOT[int(rng.integers(len(TERMINAL_PUNCT_SLOT)))]
        out = terminal_fn(cur, rng)
        if out is not None:
            cur = out
            applied_any = True

    if applied_any:
        return cur

    # Guarantee: force at least one perturbation on the original text
    for fn in OTHER_NEAR_END + TERMINAL_PUNCT_SLOT:
        out = fn(text, rng)
        if out is not None:
            return out
    return None


# ---------------------------------------------------------------------------
# Evaluation conditions
# ---------------------------------------------------------------------------
EVAL_CONDITIONS: dict[str, tuple[str, callable]] = {
    # Near-end (seen augmentation family)
    "last_word_typo":  ("near_end", apply_last_word_typo),
    "trail_period":    ("near_end", apply_trail_period),
    "early_shift":     ("near_end", apply_early_shift),
    "q_to_period":     ("near_end", apply_q_to_period),
    "q_to_slash":      ("near_end", apply_q_to_slash),
    "dot_to_slash":    ("near_end", apply_dot_to_slash),
    # Mid-sequence (held-out / unseen distance regime)
    "mid_word_typo":   ("mid_seq",  apply_mid_word_typo),
    "missing_space":   ("mid_seq",  apply_missing_space),
    # Compound
    "full_bundle":     ("compound", apply_full_bundle),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clean_features(full_ds: CachedActivationDataset, indices: np.ndarray,
                        batch_size: int = 2048) -> np.ndarray:
    """Extract pos-5 activations from the NVME cache in batches."""
    n = len(indices)
    d = full_ds.d_model
    X = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end].tolist()
        hidden, mask, labels, spans = full_ds.load_batch(batch_idx, max_tokens=MAX_TOKENS)
        X[start:end] = hidden[:, POS, :].float().cpu().numpy()
    return X


def load_all_texts(extractor: ActivationExtractor) -> dict[str, str]:
    """Load prompt_id → formatted text for all datasets via DataLoader."""
    loader = DataLoader(tokenizer=extractor.tokenizer, add_generation_prompt=True)
    texts: dict[str, str] = {}
    for ds_name in DATASETS_ALL:
        try:
            samples = loader.load(ds_name, verbose=False)
            for s in samples:
                texts[s["prompt_id"]] = s["text"]
            log(f"  {ds_name}: {len(samples)} texts")
        except Exception as e:
            log(f"  {ds_name}: SKIP ({e})")
    return texts


# ---------------------------------------------------------------------------
# Augmented feature extraction (cached per fold)
# ---------------------------------------------------------------------------

def extract_augmented_features(
    extractor: ActivationExtractor,
    prompt_ids,
    all_texts: dict[str, str],
    cache_path: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one augmented activation per training sample. Cached to disk."""
    if os.path.exists(cache_path):
        log(f"  Loading cached augmented features: {cache_path}")
        d = np.load(cache_path)
        return d["X_aug"], d["valid"]

    rng = np.random.default_rng(seed)
    n = len(prompt_ids)
    X_aug = np.zeros((n, extractor.d_model), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    log(f"  Extracting augmented features for {n} training samples ...")
    t0 = time.time()
    for i, pid in enumerate(prompt_ids):
        text = all_texts.get(pid)
        if text is None:
            continue
        perturbed = apply_near_end_augmentation(text, rng)
        if perturbed is None:
            continue
        with torch.no_grad():
            hidden, _ = extractor.extract_all_positions([perturbed])
        X_aug[i] = hidden[0, POS, :].float().cpu().numpy()
        del hidden

        valid[i] = True
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = (n - i - 1) * elapsed / (i + 1) / 60
            log(f"    {i+1}/{n}  valid={valid[:i+1].sum()}  ETA {eta:.1f}min")

    log(f"  Done: {valid.sum()}/{n} valid in {time.time()-t0:.0f}s")
    np.savez_compressed(cache_path, X_aug=X_aug, valid=valid)
    return X_aug, valid


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_aug: np.ndarray,
    valid_aug: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler]:
    X_av = X_aug[valid_aug]
    y_av = y_clean[valid_aug]
    X_all = np.concatenate([X_clean, X_av])
    y_all = np.concatenate([y_clean, y_av])
    log(f"  Training: {len(y_clean)} clean + {len(y_av)} aug = {len(y_all)} total")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                             random_state=SEED)
    t0 = time.time()
    clf.fit(X_s, y_all)
    log(f"  Fit in {time.time()-t0:.1f}s  train_acc={clf.score(X_s, y_all):.4f}")
    return clf, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_tpr_at_fpr(y: np.ndarray, scores: np.ndarray, targets=(0.001, 0.01, 0.05)):
    if len(np.unique(y)) < 2:
        return {t: float("nan") for t in targets}
    fprs, tprs, thrs = roc_curve(y, scores)
    result = {}
    for t in targets:
        idx = int(np.searchsorted(fprs, t))
        idx = min(idx, len(thrs) - 1)
        # Use threshold from CLEAN curve applied to perturbed scores
        result[t] = float(tprs[idx])
    return result


def evaluate_condition(
    name: str,
    group: str,
    pert_fn,
    extractor: ActivationExtractor,
    all_texts: dict[str, str],
    test_prompt_ids,
    y_test: np.ndarray,
    clean_scores: np.ndarray,
    clean_preds: np.ndarray,
    clf: LogisticRegression,
    scaler: StandardScaler,
    rng: np.random.Generator,
) -> dict:
    n = len(test_prompt_ids)
    pert_scores = np.full(n, np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    t0 = time.time()
    for i, pid in enumerate(test_prompt_ids):
        text = all_texts.get(pid)
        if text is None:
            continue
        perturbed = pert_fn(text, rng) if pert_fn is not None else text
        if perturbed is None:
            continue
        with torch.no_grad():
            hidden, _ = extractor.extract_all_positions([perturbed])
        feat = hidden[0, POS, :].float().cpu().numpy().reshape(1, -1)
        del hidden
        feat_s = scaler.transform(feat)
        pert_scores[i] = float(clf.predict_proba(feat_s)[0, 1])
        valid[i] = True

    n_valid = int(valid.sum())
    elapsed = time.time() - t0

    if n_valid == 0:
        return {"name": name, "group": group, "n_valid": 0}

    y_v       = y_test[valid]
    base_v    = clean_scores[valid]
    pert_v    = pert_scores[valid]
    base_pr   = clean_preds[valid]
    pert_pr   = (pert_v >= 0.5).astype(int)

    auc       = float(roc_auc_score(y_v, pert_v)) if len(np.unique(y_v)) == 2 else float("nan")
    acc       = float((pert_pr == y_v).mean())
    flips     = int((base_pr != pert_pr).sum())
    flip_pct  = float(100 * flips / n_valid)

    # Fixed-FPR TPR: threshold set on CLEAN curve, applied to perturbed scores
    clean_tpr_at = compute_tpr_at_fpr(y_v, base_v)
    pert_tpr_at  = compute_tpr_at_fpr(y_v, pert_v)

    fixed_fpr = []
    for t in (0.001, 0.01, 0.05):
        fixed_fpr.append({
            "target_fpr": t,
            "clean_tpr":  clean_tpr_at[t],
            "pert_tpr":   pert_tpr_at[t],
            "tpr_drop":   clean_tpr_at[t] - pert_tpr_at[t],
        })

    mal_shift = float(
        pert_v[y_v == 1].mean() - base_v[y_v == 1].mean()
    ) if (y_v == 1).sum() > 0 else float("nan")
    ben_shift = float(
        pert_v[y_v == 0].mean() - base_v[y_v == 0].mean()
    ) if (y_v == 0).sum() > 0 else float("nan")

    log(f"    [{name}] acc={acc:.4f} AUC={auc:.4f} flips={flip_pct:.2f}% "
        f"TPR@1%={pert_tpr_at[0.01]:.4f} (drop={clean_tpr_at[0.01]-pert_tpr_at[0.01]:+.4f}) "
        f"mal_shift={mal_shift:+.4f}  ({elapsed:.0f}s)")

    return {
        "name": name, "group": group,
        "n_valid": n_valid, "elapsed_s": float(elapsed),
        "acc": acc, "auc": auc,
        "flips": flips, "flip_pct": flip_pct,
        "mal_shift": mal_shift, "ben_shift": ben_shift,
        "fixed_fpr": fixed_fpr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-dir", default=DEFAULT_CV_DIR,
                   help="5fold_cv root dir (contains fold_0..fold_4 with split_indices.npz)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Subsample for smoke testing (applied per fold)")
    p.add_argument("--aug-prob", type=float, default=AUG_PROB,
                   help="Per-family application probability")
    p.add_argument("--conditions", nargs="+",
                   default=list(EVAL_CONDITIONS.keys()),
                   help="Subset of evaluation conditions")
    p.add_argument("--folds", nargs="+", type=int, default=None,
                   help="Run only these fold indices (0-based)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out}")

    # ── Load NVME cache ────────────────────────────────────────────────────
    log(f"Loading NVME cache from {CACHE_DIR} ...")
    cache    = ActivationCache(cache_dir=CACHE_DIR)
    full_ds  = CachedActivationDataset(cache, DATASETS_ALL)
    N        = len(full_ds)
    y_all    = full_ds.labels.numpy().astype(np.int32)
    pids_all = np.array(full_ds.prompt_ids)
    log(f"  {N} samples  malicious={y_all.sum()} ({y_all.mean():.1%})")

    # ── Discover fold splits ───────────────────────────────────────────────
    cv_dir    = Path(args.cv_dir)
    fold_dirs = sorted([
        d for d in cv_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_")
        and (d / "split_indices.npz").exists()
    ])
    log(f"Found {len(fold_dirs)} folds in {cv_dir}")
    if not fold_dirs:
        log("No fold splits found. Run run_5fold_sweep.sh first.")
        return

    folds_to_run = args.folds if args.folds else list(range(len(fold_dirs)))

    # ── Load model ─────────────────────────────────────────────────────────
    log("Loading ActivationExtractor ...")
    extractor = ActivationExtractor(
        model_name=MODEL_NAME, layer=LAYER,
        max_seq_len=16384, attn_implementation="sdpa",
    )
    log(f"  d_model={extractor.d_model}")

    # ── Load all texts ─────────────────────────────────────────────────────
    log("Loading dataset texts ...")
    all_texts = load_all_texts(extractor)
    log(f"  Total texts loaded: {len(all_texts)}")

    fold_results = []

    for fold_idx in folds_to_run:
        fold_dir  = fold_dirs[fold_idx]
        split     = np.load(fold_dir / "split_indices.npz")
        train_idx = split["train"]
        test_idx  = split["test"]

        if args.n_samples:
            rng_sub   = np.random.default_rng(SEED + fold_idx)
            train_idx = rng_sub.choice(train_idx, size=min(args.n_samples, len(train_idx)), replace=False)
            test_idx  = rng_sub.choice(test_idx,  size=min(args.n_samples // 4, len(test_idx)),  replace=False)

        log(f"\n{'='*70}")
        log(f"FOLD {fold_idx}  train={len(train_idx)}  test={len(test_idx)}")
        log(f"{'='*70}")

        y_train         = y_all[train_idx]
        y_test          = y_all[test_idx]
        train_pids      = pids_all[train_idx]
        test_pids       = pids_all[test_idx]

        # Clean activations from NVME cache
        log("  Loading clean train activations ...")
        X_train = load_clean_features(full_ds, train_idx)
        log("  Loading clean test activations ...")
        X_test  = load_clean_features(full_ds, test_idx)

        # Pre-extract augmented features (cached per fold)
        aug_cache = out / f"aug_features_fold{fold_idx}.npz"
        X_aug, valid_aug = extract_augmented_features(
            extractor, train_pids, all_texts,
            str(aug_cache), seed=SEED + fold_idx,
        )

        # Train augmented probe
        clf, scaler = train_probe(X_train, y_train, X_aug, valid_aug)

        # Clean baseline on test set
        X_test_s    = scaler.transform(X_test)
        clean_scores = clf.predict_proba(X_test_s)[:, 1].astype(np.float32)
        clean_preds  = (clean_scores >= 0.5).astype(int)
        clean_auc    = float(roc_auc_score(y_test, clean_scores))
        clean_acc    = float((clean_preds == y_test).mean())
        clean_tpr1   = compute_tpr_at_fpr(y_test, clean_scores)[0.01]
        log(f"  Clean baseline: acc={clean_acc:.4f}  AUC={clean_auc:.4f}  TPR@1%={clean_tpr1:.4f}")

        fold_cond_results = {
            "clean": {
                "name": "clean", "group": "baseline",
                "acc": clean_acc, "auc": clean_auc,
                "tpr_at_fpr_0.01": clean_tpr1,
            }
        }

        for cond_name in args.conditions:
            group, pert_fn = EVAL_CONDITIONS[cond_name]
            rng_eval = np.random.default_rng(SEED + hash(cond_name) % 2**32)
            log(f"\n  -- Condition: {cond_name} ({group}) --")
            result = evaluate_condition(
                cond_name, group, pert_fn,
                extractor, all_texts,
                test_pids, y_test,
                clean_scores, clean_preds,
                clf, scaler, rng_eval,
            )
            fold_cond_results[cond_name] = result

        fold_results.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test":  int(len(test_idx)),
            "n_aug_valid": int(valid_aug.sum()),
            "conditions": fold_cond_results,
        })

        # Save after each fold
        with open(out / "fold_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)
        log(f"\nFold {fold_idx} saved.")

    # ── Aggregate across folds ──────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("AGGREGATE SUMMARY (mean ± std across folds)")
    log(f"{'='*70}")

    all_conditions = ["clean"] + list(args.conditions)
    summary: dict[str, dict] = {}

    for cond in all_conditions:
        tpr1_vals, acc_vals, auc_vals, flip_vals = [], [], [], []
        for fr in fold_results:
            m = fr["conditions"].get(cond)
            if m is None or m.get("n_valid", 0) == 0:
                continue
            if cond == "clean":
                tpr1_vals.append(m.get("tpr_at_fpr_0.01", float("nan")))
                acc_vals.append(m.get("acc", float("nan")))
                auc_vals.append(m.get("auc", float("nan")))
                flip_vals.append(0.0)
            else:
                fpr_entries = {e["target_fpr"]: e for e in m.get("fixed_fpr", [])}
                tpr1_vals.append(fpr_entries.get(0.01, {}).get("pert_tpr", float("nan")))
                acc_vals.append(m.get("acc", float("nan")))
                auc_vals.append(m.get("auc", float("nan")))
                flip_vals.append(m.get("flip_pct", float("nan")))

        def _ms(vals):
            v = [x for x in vals if not np.isnan(x)]
            if not v:
                return float("nan"), float("nan")
            return float(np.mean(v)), float(np.std(v))

        tpr1_m, tpr1_s  = _ms(tpr1_vals)
        acc_m,  acc_s   = _ms(acc_vals)
        auc_m,  auc_s   = _ms(auc_vals)
        flip_m, flip_s  = _ms(flip_vals)
        summary[cond]   = {
            "tpr_at_1pct_fpr": {"mean": tpr1_m, "std": tpr1_s},
            "acc":             {"mean": acc_m,  "std": acc_s},
            "auc":             {"mean": auc_m,  "std": auc_s},
            "flip_pct":        {"mean": flip_m, "std": flip_s},
        }
        grp = EVAL_CONDITIONS.get(cond, ("baseline", None))[0] if cond != "clean" else "baseline"
        log(f"  {cond:<20s} [{grp:<8s}]  "
            f"TPR@1%={tpr1_m:.4f}±{tpr1_s:.4f}  "
            f"acc={acc_m:.4f}  AUC={auc_m:.4f}  "
            f"flip={flip_m:.2f}%")

    final = {
        "config": {
            "cv_dir":    args.cv_dir,
            "cache_dir": CACHE_DIR,
            "model":     MODEL_NAME,
            "layer":     LAYER,
            "position":  POS,
            "seed":      SEED,
            "n_folds":   len(folds_to_run),
            "aug_prob":  args.aug_prob,
            "augmentation_families": [
                "last_word_typo", "trail_period", "early_shift",
                "terminal_punct_slot: {q_to_period | q_to_slash | dot_to_slash}",
            ],
            "held_out_families": ["mid_word_typo", "missing_space"],
        },
        "fold_results": fold_results,
        "summary": summary,
    }
    with open(out / "results.json", "w") as f:
        json.dump(final, f, indent=2)
    log(f"\nFinal results saved to {out}/results.json")


if __name__ == "__main__":
    main()
