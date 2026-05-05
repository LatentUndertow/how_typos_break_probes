#!/usr/bin/env python3
"""
Validate LODO export integrity against the README dataset metadata.

Ground truth: the dataset class-distribution table in
`lodo_results_export/README.md` (29 datasets, n_test, n_mal, n_ben).

For each fold, we check that:
  (a) canonical `fold_<NAME>/<probe>/test_scores.npz` agrees with the
      README on n and n_mal.
  (b) `perteval_clean/fold_<NAME>_scores.npz` (if present) agrees with
      the README on n and n_mal.
For folds where (b) fails, we search the legacy
`perteval/perteval_shard_*/` and `perteval/perteval_rerun_gpu*/`
directories for a variant whose labels match the README, and prefer that.

Output:
  validated_perteval/fold_<NAME>_scores.npz  -- canonical perteval data
  validation_report.json                     -- decisions + audit log

The README explicitly recommends `perteval_clean/`, so we use it by
default and only override per-fold when integrity fails.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

EXPORT_ROOT = Path("./lodo_data/extracted/lodo_results_export")
OUT_DIR     = Path("./lodo_data/validated_perteval")
REPORT_PATH = Path(__file__).parent / "validation_report.json"

PROBES = [
    "positional_linear_pos-5",
    "mean_linear_last16",
    "mlp_all",
    "attention_all",
    "multimax_all",
]

README_TABLE_REGEX = re.compile(
    r"^\|\s*\**([A-Za-z0-9_]+)\**\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|"
)


def parse_readme_metadata(readme_path: Path) -> dict:
    """Parse the dataset-distribution table from README.md."""
    text = readme_path.read_text()
    out = {}
    for line in text.splitlines():
        m = README_TABLE_REGEX.match(line.strip())
        if not m:
            continue
        name, n, n_mal, n_ben = m.groups()
        try:
            n_i = int(n.replace(",", ""))
            mal_i = int(n_mal.replace(",", ""))
            ben_i = int(n_ben.replace(",", ""))
        except ValueError:
            continue
        if mal_i + ben_i != n_i:
            continue
        out[name] = {"n": n_i, "n_mal": mal_i, "n_ben": ben_i}
    return out


def load_labels(npz_path: Path) -> tuple[int, int]:
    d = np.load(npz_path)
    if "labels" not in d:
        return -1, -1
    lbls = d["labels"]
    return int(len(lbls)), int(lbls.sum())


def find_matching_perteval_variant(
    fold_name: str, expected_n: int, expected_mal: int, export_root: Path
) -> tuple[Path | None, list[dict]]:
    """Search legacy shards/reruns for a variant whose (n, n_mal) match README."""
    candidates: list[dict] = []
    legacy_root = export_root / "perteval"
    if not legacy_root.exists():
        return None, candidates

    for sub in sorted(legacy_root.iterdir()):
        if not sub.is_dir():
            continue
        cand = sub / f"fold_{fold_name}_scores.npz"
        if not cand.exists():
            continue
        n, mal = load_labels(cand)
        candidates.append(
            {
                "path": str(cand),
                "source": sub.name,
                "n": n,
                "n_mal": mal,
                "matches_readme": (n == expected_n and mal == expected_mal),
            }
        )
    matches = [c for c in candidates if c["matches_readme"]]
    if not matches:
        return None, candidates
    # Prefer reruns over original shards (reruns are the corrected data)
    rerun_matches = [c for c in matches if c["source"].startswith("perteval_rerun")]
    chosen = rerun_matches[0] if rerun_matches else matches[0]
    return Path(chosen["path"]), candidates


def main() -> int:
    if not EXPORT_ROOT.exists():
        print(f"ERROR: extract root not found: {EXPORT_ROOT}", file=sys.stderr)
        print("Run 00_extract.sh first.", file=sys.stderr)
        return 1

    readme_path = EXPORT_ROOT / "README.md"
    if not readme_path.exists():
        print(f"ERROR: README.md not found at {readme_path}", file=sys.stderr)
        return 1

    metadata = parse_readme_metadata(readme_path)
    print(f"[01_validate] Parsed README: {len(metadata)} datasets")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    report = {
        "export_root": str(EXPORT_ROOT),
        "n_datasets_in_readme": len(metadata),
        "folds": {},
    }
    n_canon_ok = 0
    n_perteval_clean_ok = 0
    n_perteval_overridden = 0
    n_perteval_unrecoverable = 0
    n_perteval_unavailable = 0

    for fold_name, expected in sorted(metadata.items()):
        entry: dict = {
            "expected": expected,
            "canonical_check": {},
            "perteval_clean_check": {},
            "decision": {},
        }

        # (a) canonical test_scores.npz check
        canon_path = (
            EXPORT_ROOT / f"fold_{fold_name}" / PROBES[0] / "test_scores.npz"
        )
        if canon_path.exists():
            n, mal = load_labels(canon_path)
            entry["canonical_check"] = {
                "path": str(canon_path),
                "n": n,
                "n_mal": mal,
                "matches_readme": (n == expected["n"] and mal == expected["n_mal"]),
            }
            if entry["canonical_check"]["matches_readme"]:
                n_canon_ok += 1
        else:
            entry["canonical_check"] = {"path": str(canon_path), "missing": True}

        # (b) perteval_clean check
        pc_path = EXPORT_ROOT / "perteval_clean" / f"fold_{fold_name}_scores.npz"
        if pc_path.exists():
            n, mal = load_labels(pc_path)
            entry["perteval_clean_check"] = {
                "path": str(pc_path),
                "n": n,
                "n_mal": mal,
                "matches_readme": (n == expected["n"] and mal == expected["n_mal"]),
            }
            if entry["perteval_clean_check"]["matches_readme"]:
                # Use perteval_clean as-is.
                shutil.copy2(pc_path, OUT_DIR / pc_path.name)
                entry["decision"] = {"source": "perteval_clean", "path": str(pc_path)}
                n_perteval_clean_ok += 1
            else:
                # Search legacy variants.
                chosen, candidates = find_matching_perteval_variant(
                    fold_name, expected["n"], expected["n_mal"], EXPORT_ROOT
                )
                entry["perteval_clean_check"]["legacy_candidates"] = candidates
                if chosen is not None:
                    shutil.copy2(chosen, OUT_DIR / pc_path.name)
                    entry["decision"] = {
                        "source": "legacy_override",
                        "path": str(chosen),
                    }
                    n_perteval_overridden += 1
                else:
                    entry["decision"] = {"source": "unrecoverable"}
                    n_perteval_unrecoverable += 1
        else:
            entry["perteval_clean_check"] = {"path": str(pc_path), "missing": True}
            entry["decision"] = {"source": "unavailable"}
            n_perteval_unavailable += 1

        report["folds"][fold_name] = entry

    report["summary"] = {
        "canonical_ok": n_canon_ok,
        "perteval_clean_ok": n_perteval_clean_ok,
        "perteval_overridden_with_legacy": n_perteval_overridden,
        "perteval_unrecoverable": n_perteval_unrecoverable,
        "perteval_unavailable": n_perteval_unavailable,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    # Console summary
    print("\n=== Summary ===")
    print(f"  canonical test_scores ok:        {n_canon_ok}/{len(metadata)}")
    print(f"  perteval_clean ok:               {n_perteval_clean_ok}")
    print(f"  perteval overridden from legacy: {n_perteval_overridden}")
    print(f"  perteval unrecoverable:          {n_perteval_unrecoverable}")
    print(f"  perteval unavailable:            {n_perteval_unavailable}")

    if n_perteval_overridden:
        print("\n  Overrides applied:")
        for f, e in report["folds"].items():
            if e["decision"]["source"] == "legacy_override":
                src = e["decision"]["path"].split("/perteval/")[-1]
                print(f"    {f:<32s} <- {src}")

    if n_perteval_unrecoverable:
        print("\n  Unrecoverable folds (no variant matched README):")
        for f, e in report["folds"].items():
            if e["decision"]["source"] == "unrecoverable":
                print(f"    {f}")

    print(f"\n  Validated perteval written to: {OUT_DIR}")
    print(f"  Full report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
