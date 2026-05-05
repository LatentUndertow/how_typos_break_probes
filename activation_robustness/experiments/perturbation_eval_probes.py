#!/usr/bin/env python3
"""
Perturbation evaluation for trained probe architectures.

Loads trained probes from the sweep, loads Llama for on-the-fly extraction,
applies perturbations to test samples, and scores with all probes.

Uses the online_probes infra:
- ActivationExtractor for model loading + extraction
- DataLoader for text loading (with same chat template formatting)
- MultiArchProbe for scoring

Output per probe: (clean_score, perturbed_score, label, prompt_id) per test sample.

Usage:
    python perturbation_eval_probes.py
    python perturbation_eval_probes.py --n-test 1000  # quick test
    python perturbation_eval_probes.py --conditions full_bundle q_to_period
"""
import sys
import os
import re
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
SWEEP_DIR = _REPO / "activation_robustness" / "results" / "probe_architecture_sweep_29ds"
OUTPUT_DIR = _REPO / "activation_robustness" / "results" / "perturbation_eval_probes_29ds"

# NOTE: DATASET_META was added to the dataset library AFTER this cache was built
# (cache 2026-03-15, DATASET_META PR 2026-03-22). For 3 datasets the DATASET_META
# defaults disagree with the class __init__ defaults that were in force at cache
# build time. We pin the cache-era defaults explicitly so online extraction
# reproduces the cached text (cosine ~0.9995 vs cache for all datasets). See
# activation_robustness/docs/08_paper_plan.md for details.
DATASETS = [
    {"class": "EnronDataset", "params": {"include_email_format": True}},     # cache-era default
    "Dolly15kDataset",
    {"class": "OpenOrcaDataset", "params": {"include_system_prompt": True}}, # cache-era default
    "PromptsRanked10kDataset",
    "AlpacaDataset", "SafeGuardDataset", "QualifireDataset", "SoftAgeDataset",
    {"class": "BIPIADataset", "params": {"bipia_root": "./data/BIPIA",
                                          "task_names": ["email", "code", "table"]}},
    {"class": "InjecAgentDataset", "params": {"injecagent_root": "./data/InjecAgent",
                                               "attack_types": ["dh", "ds"], "setting": "base"}},
    {"class": "LLMailDataset", "params": {"include_email_format": True}},    # cache-era default
    "MosscapDataset",
    "WildJailbreakDataset", "JayavibhavDataset", "DeepsetDataset",
    "YanismiraouiDataset", "AdvBenchDataset",
    # 12 datasets added in the 29-dataset sweep expansion
    "HarmBenchDataset", "AgentDojoDataset", "APIGenMTDataset",
    "BitextCustomerSupportDataset", "CodeExerciseDataset",
    "GandalfSummarizationDataset", "JailbreakClassificationDataset",
    "PythonCodeAlpacaDataset", "PythonCodes25kDataset",
    {"class": "ScamDataset", "params": {"scam_root": "./data/SCAM"}},
    "WritingPromptsDataset", "XlamFunctionCallingDataset",
]

# Dataset name extraction helper
def _ds_name(entry):
    if isinstance(entry, str):
        return entry
    return next(iter(entry)) if isinstance(entry, dict) and "class" not in entry else entry["class"]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 31
SEED = 42

# ---------------------------------------------------------------------------
# Perturbation functions (from perturbation_decomposition_80_20.py)
# ---------------------------------------------------------------------------

USER_MSG_PATTERN = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
    re.DOTALL,
)

QWERTY_ADJACENT = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etf", "t": "ryg",
    "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "o",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
    "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "iolmj",
    "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_user_span(input_text):
    m = USER_MSG_PATTERN.search(input_text)
    if m is None:
        return None, None, None
    return m.start(1), m.end(1), m.group(1)


def _replace_user_content(input_text, cs, ce, new_content):
    return input_text[:cs] + new_content + input_text[ce:]


def _typo_in_word(word_text, rng):
    candidates = [i for i, c in enumerate(word_text) if c.lower() in QWERTY_ADJACENT]
    if not candidates:
        return None
    idx = int(rng.choice(candidates))
    char = word_text[idx]
    adj = QWERTY_ADJACENT[char.lower()]
    replacement = adj[rng.integers(len(adj))]
    if char.isupper():
        replacement = replacement.upper()
    return word_text[:idx] + replacement + word_text[idx + 1:]


def apply_q_to_period(input_text, rng):
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    stripped = user.rstrip()
    if not stripped.endswith("?"):
        return None
    trailing_ws = user[len(stripped):]
    new = stripped[:-1] + "." + trailing_ws
    return _replace_user_content(input_text, cs, ce, new)


def apply_last_word_typo(input_text, rng):
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", user)]
    if not word_spans:
        return None
    ws, we = word_spans[-1]
    new_word = _typo_in_word(user[ws:we], rng)
    if new_word is None:
        return None
    new = user[:ws] + new_word + user[we:]
    return _replace_user_content(input_text, cs, ce, new)


def apply_mid_word_typo(input_text, rng):
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", user)]
    if len(word_spans) < 4:
        return None
    eligible = word_spans[:-3]
    pick_idx = int(rng.integers(len(eligible)))
    ws, we = eligible[pick_idx]
    new_word = _typo_in_word(user[ws:we], rng)
    if new_word is None:
        return None
    new = user[:ws] + new_word + user[we:]
    return _replace_user_content(input_text, cs, ce, new)


def apply_early_shift(input_text, rng):
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    m = re.search(r"[a-z][!?]", user)
    if m is None:
        return None
    pos = m.start()
    new = user[:pos] + user[pos].upper() + user[pos + 1:]
    return _replace_user_content(input_text, cs, ce, new)


def apply_trail_period(input_text, rng):
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    rstripped = user.rstrip()
    trailing_ws = user[len(rstripped):]
    if rstripped.endswith("."):
        new = rstripped[:-1] + trailing_ws
    elif rstripped and rstripped[-1] not in "?!,;:":
        new = rstripped + "." + trailing_ws
    else:
        return None
    if new == user:
        return None
    return _replace_user_content(input_text, cs, ce, new)


def apply_full_bundle(input_text, rng):
    current = input_text
    any_applied = False
    for fn in (apply_q_to_period, apply_last_word_typo, apply_mid_word_typo,
               apply_early_shift, apply_trail_period):
        out = fn(current, rng)
        if out is not None:
            current = out
            any_applied = True
    return current if any_applied else None


def apply_sloppy_typist(input_text, rng, edits_per_n_tokens=40):
    """Realistic sloppy typist: scattered errors throughout + trailing period.

    Scattered error types (randomly chosen per site):
      - adjacent_key: QWERTY neighbor substitution
      - drop_char: delete one character (missed key)
      - double_space: insert extra space after a space
      - missing_space: remove a space (join two words)

    Rate: ~1 error per edits_per_n_tokens tokens in the user message.
    Plus trailing period toggle at the end.
    """
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None

    # Estimate token count from character count (~4 chars per token)
    est_tokens = len(user) / 4
    n_scattered = max(1, int(est_tokens / edits_per_n_tokens))

    # Collect candidate positions for each error type
    alpha_positions = [i for i, c in enumerate(user) if c.isalpha() and c.lower() in QWERTY_ADJACENT]
    space_positions = [i for i, c in enumerate(user) if c == ' ' and i > 0 and i < len(user) - 1]

    # Build list of editable characters (any of the 4 types)
    candidates = []
    for i in alpha_positions:
        candidates.append((i, 'adjacent_key'))
        candidates.append((i, 'drop_char'))
    for i in space_positions:
        candidates.append((i, 'double_space'))
        candidates.append((i, 'missing_space'))

    if not candidates:
        return None

    # Spread edits evenly: divide user text into n_scattered zones, pick one per zone
    user_len = len(user)
    zone_size = max(1, user_len // n_scattered)
    edits = []  # list of (position, edit_type)
    used_positions = set()

    for zone_start in range(0, user_len, zone_size):
        if len(edits) >= n_scattered:
            break
        zone_end = min(zone_start + zone_size, user_len)
        zone_cands = [(pos, typ) for pos, typ in candidates
                      if zone_start <= pos < zone_end and pos not in used_positions]
        if not zone_cands:
            continue
        pick = zone_cands[rng.integers(len(zone_cands))]
        edits.append(pick)
        used_positions.add(pick[0])

    if not edits:
        return None

    # Apply edits in reverse order (so positions don't shift)
    new_user = list(user)
    for pos, edit_type in sorted(edits, key=lambda x: x[0], reverse=True):
        if edit_type == 'adjacent_key':
            char = new_user[pos]
            adj = QWERTY_ADJACENT[char.lower()]
            rep = adj[rng.integers(len(adj))]
            if char.isupper():
                rep = rep.upper()
            new_user[pos] = rep
        elif edit_type == 'drop_char':
            del new_user[pos]
        elif edit_type == 'double_space':
            new_user.insert(pos, ' ')
        elif edit_type == 'missing_space':
            del new_user[pos]

    new_user_str = ''.join(new_user)

    # Trailing period toggle
    rstripped = new_user_str.rstrip()
    trailing_ws = new_user_str[len(rstripped):]
    if rstripped.endswith("."):
        new_user_str = rstripped[:-1] + trailing_ws
    elif rstripped and rstripped[-1] not in "?!,;:":
        new_user_str = rstripped + "." + trailing_ws

    if new_user_str == user:
        return None

    return _replace_user_content(input_text, cs, ce, new_user_str)


def apply_every_second_word(input_text, rng):
    """Aggressive: typo every 2nd word (words at index 1, 3, 5, ...).

    Each targeted word gets one QWERTY-adjacent letter substitution via
    _typo_in_word. Processes in reverse order so positions remain valid
    across replacements. About half the words are perturbed.
    """
    cs, ce, user = find_user_span(input_text)
    if user is None:
        return None
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", user)]
    if len(word_spans) < 2:
        return None
    target_idx = list(range(1, len(word_spans), 2))
    new_user = user
    applied = 0
    for idx in sorted(target_idx, reverse=True):
        ws, we = word_spans[idx]
        new_word = _typo_in_word(new_user[ws:we], rng)
        if new_word is not None:
            new_user = new_user[:ws] + new_word + new_user[we:]
            applied += 1
    if applied == 0 or new_user == user:
        return None
    return _replace_user_content(input_text, cs, ce, new_user)


PERTURBATION_FNS = {
    "q_to_period": apply_q_to_period,
    "last_word_typo": apply_last_word_typo,
    "mid_word_typo": apply_mid_word_typo,
    "early_shift": apply_early_shift,
    "trail_period": apply_trail_period,
    "full_bundle": apply_full_bundle,
    "sloppy_typist": apply_sloppy_typist,
    "every_second_word": apply_every_second_word,
}


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probe(probe_dir, extractor, device):
    """Load a trained MultiArchProbe + its scaler from a sweep dir.

    Requires `probe.pt`, `config.json`, and (if normalize='standard') `scaler.pt`.
    Scaler is saved by the patched `probe_architecture_sweep.py` as:
        {"mean": Tensor(D,), "std": Tensor(D,)}  # training-time StandardScaler stats

    Attaches `clf._eval_mean` / `clf._eval_std` GPU tensors for in-loop normalization
    (None when normalize='none').
    """
    config_path = probe_dir / "config.json"
    probe_path = probe_dir / "probe.pt"

    if not config_path.exists() or not probe_path.exists():
        return None

    with open(config_path) as f:
        params = json.load(f)

    # _eval_slice is a [start, stop] pair for slicing online hidden at eval
    # time, mirroring how the probe's training data was sliced. This is the
    # single source of truth for input windowing — supersedes _max_tokens
    # (still read for backward compat).
    eval_slice = params.pop("_eval_slice", None)
    max_tokens = params.pop("_max_tokens", None)
    if eval_slice is None and max_tokens is not None:
        # Back-compat: old configs without _eval_slice
        eval_slice = [-max_tokens, None]

    config = ProbeConfig(**params)
    clf = MultiArchProbe(config=config, device=device, extractor=extractor)

    # Build model and load weights
    clf._d_model = extractor.d_model
    clf._model = clf._build_model(clf._d_model).to(device)
    state = torch.load(probe_path, map_location=device, weights_only=True)
    clf._model.load_state_dict(state)
    clf._model.eval()
    clf._fitted = True
    clf._eval_slice = eval_slice  # [start, stop] or None

    # Attach normalization stats
    if config.normalize == "standard":
        scaler_path = probe_dir / "scaler.pt"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Probe {probe_dir.name} was trained with normalize='standard' but "
                f"{scaler_path} is missing. Retrain with the patched sweep script."
            )
        scaler = torch.load(scaler_path, map_location=device, weights_only=True)
        clf._eval_mean = scaler["mean"].to(device).float()
        clf._eval_std = scaler["std"].to(device).float()
    elif config.normalize == "none":
        clf._eval_mean = None
        clf._eval_std = None
    else:
        raise NotImplementedError(
            f"normalize='{config.normalize}' not handled in perturbation eval load_probe"
        )

    return clf


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_test_texts(extractor, datasets):
    """Load formatted text for test samples by replaying the DataLoader.

    Returns dict mapping prompt_id -> formatted_text.
    """
    log("Loading test sample texts via DataLoader...")
    loader = DataLoader(
        tokenizer=extractor.tokenizer,
        add_generation_prompt=True,
    )

    all_texts = {}
    for entry in datasets:
        ds_name = _ds_name(entry)
        spec = entry
        try:
            samples = loader.load(spec, verbose=False)
        except Exception as e:
            log(f"  {ds_name}: SKIP (load error: {e})")
            continue
        for s in samples:
            all_texts[s["prompt_id"]] = s["text"]
        log(f"  {ds_name}: {len(samples)} samples loaded")

    return all_texts


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(extractor, clean_text, cache_ds, sample_idx):
    """Verify that online extraction matches cached activations for a sample."""
    # Extract online
    hidden, _ = extractor.extract_all_positions([clean_text])
    online_act = hidden[0, -5, :].float().cpu()  # pos -5

    # Load from cache
    ds_idx = int(cache_ds._dataset_indices[sample_idx])
    off = int(cache_ds.offsets[sample_idx])
    length = int(cache_ds.lengths[sample_idx])
    cached_act = cache_ds._mmaps[ds_idx][off + length - 5].float().cpu()

    cos = torch.nn.functional.cosine_similarity(
        online_act.unsqueeze(0), cached_act.unsqueeze(0)
    ).item()
    return cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=None,
                        help="Limit test samples (default: all)")
    parser.add_argument("--conditions", nargs="+", default=["full_bundle"],
                        help="Perturbation conditions to evaluate")
    parser.add_argument("--probe-dirs", nargs="+", default=None,
                        help="Probe directory names relative to --sweep-dir (default: auto-discover)")
    parser.add_argument("--sweep-dir", type=str, default=None,
                        help="Directory containing trained probes and split_indices.npz "
                             "(default: probe_architecture_sweep/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for scores (default: perturbation_eval_probes/)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Extraction batch size (1=sequential, no padding effects)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else SWEEP_DIR
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split
    split = np.load(sweep_dir / "split_indices.npz")
    test_idx = split["test"]
    log(f"Test split: {len(test_idx)} samples")

    if args.n_test:
        test_idx = test_idx[:args.n_test]
        log(f"  Limiting to first {args.n_test}")

    # Load extractor (loads Llama)
    log("Loading model...")
    extractor = ActivationExtractor(
        model_name=MODEL_NAME,
        layer=LAYER,
        max_seq_len=16384,
        attn_implementation="sdpa",
    )
    log("Model loaded")

    # Load cache for prompt_ids and sanity check
    cache = ActivationCache(cache_dir=CACHE_DIR)
    dataset_names = [_ds_name(d) for d in DATASETS]
    cache_ds = CachedActivationDataset(cache, dataset_names)
    prompt_ids = [cache_ds.prompt_ids[i] for i in test_idx]
    labels = cache_ds.labels[test_idx].numpy()
    log(f"Test: {int(labels.sum())} malicious, {int(len(labels) - labels.sum())} benign")

    # Load texts
    all_texts = load_test_texts(extractor, DATASETS)

    # Map test samples to their formatted text
    test_texts = []
    valid_mask = []
    for pid in prompt_ids:
        if pid in all_texts:
            test_texts.append(all_texts[pid])
            valid_mask.append(True)
        else:
            test_texts.append(None)
            valid_mask.append(False)
    valid_mask = np.array(valid_mask)
    n_valid = valid_mask.sum()
    log(f"Matched {n_valid}/{len(test_idx)} test samples to text ({len(test_idx) - n_valid} missing)")

    # Sanity check: verify online extraction matches cache for a few samples
    log("Sanity check: online extraction vs cache...")
    for i in range(min(5, n_valid)):
        idx = np.where(valid_mask)[0][i]
        cos = sanity_check(extractor, test_texts[idx], cache_ds, test_idx[idx])
        log(f"  Sample {idx}: cosine={cos:.6f}")
        if cos < 0.99:
            log(f"  WARNING: low cosine similarity — extraction mismatch!")

    # Discover trained probes
    if args.probe_dirs:
        probe_names = args.probe_dirs
    else:
        probe_names = [
            d.name for d in sweep_dir.iterdir()
            if d.is_dir() and (d / "probe.pt").exists()
        ]
    probe_names = sorted(probe_names)
    log(f"Found {len(probe_names)} trained probes: {probe_names}")

    # Load probes + their training-time scalers (saved per-probe by the sweep script).
    probes = {}
    for name in probe_names:
        clf = load_probe(sweep_dir / name, extractor, device)
        if clf is not None:
            probes[name] = clf
            log(f"  Loaded {name} (normalize={clf.config.normalize})")
        else:
            log(f"  SKIP {name} (missing files)")

    # ---------------------------------------------------------------------------
    # Sample-by-sample scoring: extract once per text, score all probes
    # No batching → no padding effects on activations
    # ---------------------------------------------------------------------------
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Prepare perturbation conditions
    conditions = {}
    for cond_name in args.conditions:
        if cond_name not in PERTURBATION_FNS:
            log(f"Unknown condition: {cond_name}, skipping")
            continue
        conditions[cond_name] = PERTURBATION_FNS[cond_name]

    # Pre-apply perturbations (text-level, no model needed)
    log(f"\nApplying perturbations ({list(conditions.keys())})...")
    perturbed_texts = {}  # cond_name -> list aligned with test_texts
    for cond_name, perturb_fn in conditions.items():
        pert_list = []
        n_applied = 0
        for text, v in zip(test_texts, valid_mask):
            if not v:
                pert_list.append(None)
                continue
            pert = perturb_fn(text, rng)
            pert_list.append(pert)
            if pert is not None:
                n_applied += 1
        perturbed_texts[cond_name] = pert_list
        log(f"  {cond_name}: applied to {n_applied}/{n_valid} samples")

    # Score sample-by-sample: one extraction per text, all probes score it
    n_total = len(test_idx)
    probe_names = list(probes.keys())

    # Initialize score arrays
    clean_scores = {name: np.full(n_total, np.nan) for name in probe_names}
    pert_scores = {
        cond: {name: np.full(n_total, np.nan) for name in probe_names}
        for cond in conditions
    }

    log(f"\nScoring {n_total} test samples (batch_size=1, sequential)...")
    t0 = time.time()

    for i in range(n_total):
        if not valid_mask[i]:
            continue

        clean_text = test_texts[i]

        # Extract clean activations (batch_size=1, no padding)
        with torch.no_grad():
            hidden_clean, _ = extractor.extract_all_positions([clean_text])
            # hidden_clean: (1, T, D) on GPU
            hidden_clean_f = hidden_clean.float()

            # Score all probes on clean (applying per-probe windowing + normalization)
            for name, clf in probes.items():
                clf._model.eval()
                if clf._eval_slice is not None:
                    s, e = clf._eval_slice
                    x = hidden_clean_f[:, s:e, :]
                else:
                    x = hidden_clean_f
                if clf._eval_mean is not None:
                    x = (x - clf._eval_mean) / clf._eval_std
                logits = clf._model(x)
                score = torch.sigmoid(logits).cpu().item()
                clean_scores[name][i] = score

            del hidden_clean, hidden_clean_f

        # Extract and score perturbed versions
        for cond_name in conditions:
            pert_text = perturbed_texts[cond_name][i]
            if pert_text is None:
                continue

            with torch.no_grad():
                hidden_pert, _ = extractor.extract_all_positions([pert_text])
                hidden_pert_f = hidden_pert.float()

                for name, clf in probes.items():
                    clf._model.eval()
                    if clf._eval_slice is not None:
                        s, e = clf._eval_slice
                        x = hidden_pert_f[:, s:e, :]
                    else:
                        x = hidden_pert_f
                    if clf._eval_mean is not None:
                        x = (x - clf._eval_mean) / clf._eval_std
                    logits = clf._model(x)
                    score = torch.sigmoid(logits).cpu().item()
                    pert_scores[cond_name][name][i] = score

                del hidden_pert, hidden_pert_f

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate / 60
            log(f"  [{i+1}/{n_total}] {rate:.1f} samples/sec, ETA {eta:.0f} min")

    total_time = time.time() - t0
    log(f"Scoring done in {total_time:.0f}s ({n_total / total_time:.1f} samples/sec)")

    # Compute and report metrics per condition per probe
    for cond_name in conditions:
        log(f"\n{'='*60}")
        log(f"  Condition: {cond_name}")
        log(f"{'='*60}")

        for name in probe_names:
            # Samples where both clean and perturbed scores exist
            both_valid_i = ~np.isnan(clean_scores[name]) & ~np.isnan(pert_scores[cond_name][name])
            if both_valid_i.sum() == 0:
                log(f"  {name}: no valid samples")
                continue

            c_scores = clean_scores[name][both_valid_i]
            p_scores = pert_scores[cond_name][name][both_valid_i]
            c_labels = labels[both_valid_i]

            clean_preds = (c_scores >= 0.5).astype(int)
            pert_preds = (p_scores >= 0.5).astype(int)
            flips = (clean_preds != pert_preds).sum()
            flip_rate = flips / len(c_scores)

            score_shift = p_scores - c_scores
            mal = c_labels == 1
            ben = c_labels == 0
            mal_shift = score_shift[mal].mean() if mal.any() else 0
            ben_shift = score_shift[ben].mean() if ben.any() else 0

            clean_acc = accuracy_score(c_labels, clean_preds)
            pert_acc = accuracy_score(c_labels, pert_preds)
            clean_auc = roc_auc_score(c_labels, c_scores) if len(np.unique(c_labels)) > 1 else 0
            pert_auc = roc_auc_score(c_labels, p_scores) if len(np.unique(c_labels)) > 1 else 0

            log(f"  {name} (n={both_valid_i.sum()}):")
            log(f"    Clean: acc={clean_acc:.4f}, AUC={clean_auc:.4f}")
            log(f"    Pert:  acc={pert_acc:.4f}, AUC={pert_auc:.4f}")
            log(f"    Flip rate: {flip_rate:.4f} ({flips}/{len(c_scores)})")
            log(f"    Score shift: mal={mal_shift:+.4f}, ben={ben_shift:+.4f}")

    # Save all scores
    save_data = {
        "labels": labels,
        "valid_mask": valid_mask,
        "prompt_ids": np.array(prompt_ids, dtype=object),
    }
    for name in probe_names:
        save_data[f"{name}_clean"] = clean_scores[name]
        for cond_name in conditions:
            save_data[f"{name}_{cond_name}"] = pert_scores[cond_name][name]

    np.savez_compressed(out_dir / "all_scores.npz", **save_data)
    log(f"\nSaved all scores to {out_dir / 'all_scores.npz'}")

    log("\nDone.")


if __name__ == "__main__":
    main()
