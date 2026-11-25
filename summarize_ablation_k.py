#!/usr/bin/env python
"""Collect the metrics.json of an ablation_k sweep into one readable folder.

For every all_trials directory given (or found under --inference-root), writes a
`summary/` holding one metrics_bk_<k>.json per trial plus the comparison tables
the paper needs. Re-running overwrites the summary; the trials are only read.

    python summarize_ablation_k.py                       # every dataset found
    python summarize_ablation_k.py --dataset brats       # just this one
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_INFERENCE_ROOT = Path("/home/guibo/links/scratch/inference/unet")

# (column, path in metrics.json). Kept explicit so the tables stay stable even
# when a run carries extra keys.
SEGMENTATION_COLUMNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("dice", ("segmentation", "dice")),
    ("iou", ("segmentation", "iou")),
    ("pixel_acc", ("segmentation", "pixel_acc")),
    ("hd95", ("segmentation", "hd95")),
    ("dice_identified_correct", ("segmentation", "dice_identified_correct")),
    ("energy_thr", ("segmentation", "energy_below_threshold", "threshold")),
    ("energy_thr_source", ("segmentation", "energy_below_threshold", "threshold_source")),
    ("energy_filtered_dice", ("segmentation", "energy_below_threshold", "dice")),
    ("energy_filtered_iou", ("segmentation", "energy_below_threshold", "iou")),
    ("num_pixels", ("num_pixels",)),
    ("num_error_pixels", ("num_error_pixels",)),
]

DETECTION_COLUMNS = ("auroc", "aupr", "fpr95")
FILTERED_COLUMNS = ("threshold", "threshold_source", "achieved_tpr", "frac_pixels", "dice", "iou", "pixel_acc")


def dig(payload: dict, path: Tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def boundary_k_of(trial_dir: Path) -> int:
    """boundary_k from grid_trial.json, falling back to the _bk_<k>_ in the name."""
    grid_trial = trial_dir / "grid_trial.json"
    if grid_trial.is_file():
        params = json.loads(grid_trial.read_text()).get("params", {})
        if "boundary_k" in params:
            return int(params["boundary_k"])
    parts = trial_dir.name.split("_")
    if "bk" in parts:
        return int(parts[parts.index("bk") + 1])
    raise ValueError(f"cannot tell boundary_k for {trial_dir}")


def collect(all_trials: Path) -> List[Tuple[int, Path, dict]]:
    trials = []
    for trial_dir in sorted(all_trials.glob("trial_*")):
        metrics_path = trial_dir / "metrics.json"
        if not metrics_path.is_file():
            print(f"  [skip] {trial_dir.name}: no metrics.json (still running or crashed)")
            continue
        trials.append((boundary_k_of(trial_dir), metrics_path, json.loads(metrics_path.read_text())))
    return sorted(trials, key=lambda item: item[0])


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def markdown_table(header: List[str], rows: List[List[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def summarize(all_trials: Path, dataset: str) -> bool:
    trials = collect(all_trials)
    if not trials:
        print(f"  nothing to summarize in {all_trials}")
        return False

    summary_dir = all_trials / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for stale in summary_dir.glob("metrics_bk_*.json"):
        stale.unlink()

    for k, metrics_path, _ in trials:
        shutil.copy2(metrics_path, summary_dir / f"metrics_bk_{k}.json")

    # 1. segmentation, one row per k
    seg_header = ["boundary_k"] + [name for name, _ in SEGMENTATION_COLUMNS]
    seg_rows = [[k] + [dig(payload, path) for _, path in SEGMENTATION_COLUMNS] for k, _, payload in trials]
    write_csv(summary_dir / "comparison_segmentation.csv", seg_header, seg_rows)

    # 2. error detection, one row per (k, score)
    score_names = sorted({name for _, _, payload in trials for name in payload.get("uncertainty", {})})
    det_header = ["boundary_k", "score", *DETECTION_COLUMNS]
    det_rows = [
        [k, name] + [payload["uncertainty"][name].get(column) for column in DETECTION_COLUMNS]
        for k, _, payload in trials
        for name in score_names
        if name in payload.get("uncertainty", {})
    ]
    write_csv(summary_dir / "comparison_detection.csv", det_header, det_rows)

    # 3. segmentation of the pixels each score keeps, one row per (k, score)
    flt_header = ["boundary_k", "score", *FILTERED_COLUMNS]
    flt_rows = [
        [k, name] + [block.get(column) for column in FILTERED_COLUMNS]
        for k, _, payload in trials
        for name, block in payload.get("segmentation", {}).get("score_thresholds", {}).items()
    ]
    if flt_rows:
        write_csv(summary_dir / "comparison_filtered_segmentation.csv", flt_header, flt_rows)

    # 4. the same three tables, readable
    sections = [
        f"# ablation boundary_k - {dataset}",
        "",
        f"{len(trials)} trials, k = {', '.join(str(k) for k, _, _ in trials)}.",
        "Source: " + str(all_trials),
        "",
        "## Segmentation",
        "",
        markdown_table(seg_header, seg_rows),
        "",
        "## Error detection",
        "",
        markdown_table(det_header, det_rows),
    ]
    if flt_rows:
        sections += [
            "",
            "## Segmentation of the kept pixels, per score",
            "",
            "`threshold_source: tpr>=0.95` means the score was cut where it flags 95% of",
            "the error pixels, the operating point fpr95 describes. `frac_pixels` is the",
            "share of pixels kept.",
            "",
            markdown_table(flt_header, flt_rows),
        ]
    (summary_dir / "comparison.md").write_text("\n".join(sections) + "\n", encoding="utf-8")

    print(f"  wrote {summary_dir} ({len(trials)} trials)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inference-root", type=Path, default=DEFAULT_INFERENCE_ROOT)
    parser.add_argument("--sweep", default="ablation_k", help="Sweep directory under <root>/<dataset>/")
    parser.add_argument("--dataset", action="append", default=[], help="Restrict to these datasets. Repeatable.")
    parser.add_argument("--skip", action="append", default=[], help="Datasets to leave alone. Repeatable.")
    args = parser.parse_args()

    candidates = sorted(args.inference_root.glob(f"*/{args.sweep}/all_trials"))
    written = 0
    for all_trials in candidates:
        dataset = all_trials.parent.parent.name
        if args.dataset and dataset not in args.dataset:
            continue
        if dataset in args.skip:
            print(f"{dataset}: skipped on request")
            continue
        print(f"{dataset}:")
        written += summarize(all_trials, dataset)
    print(f"\n{written} summary folder(s) written.")


if __name__ == "__main__":
    main()
