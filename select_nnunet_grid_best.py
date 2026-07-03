import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the best nnU-Net EBO grid-search trial by min val loss, then min energy FPR95."
    )
    parser.add_argument("--results-base", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Only consider checkpoints from this fold.")
    return parser


def _finite_or_inf(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.inf
    return numeric if math.isfinite(numeric) else math.inf


def trial_key(trial: dict[str, Any]) -> tuple[float, float, str]:
    return (
        _finite_or_inf(trial.get("best_val_energy_fpr95")),
        _finite_or_inf(trial.get("best_val_loss")),
        trial.get("trial_dir", ""),
    )


def main() -> None:
    args = build_argparser().parse_args()
    summaries = sorted(args.results_base.glob("trial_*/**/ebo_selection_summary.json"))
    trials = []

    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary_path.parent / summary.get("best_checkpoint", "checkpoint_best.pth")
        if not checkpoint.is_file():
            continue
        if args.fold is not None and summary_path.parent.name != f"fold_{args.fold}":
            continue
        trials.append(
            {
                "trial_name": summary_path.relative_to(args.results_base).parts[0],
                "trial_dir": str(summary_path.parent),
                "checkpoint": str(checkpoint),
                "best_val_loss": summary.get("best_val_loss"),
                "best_val_energy_fpr95": summary.get("best_val_energy_fpr95"),
                "selection_rule": summary.get("selection_rule", "min_val_loss_then_min_val_energy_fpr95"),
            }
        )

    if not trials:
        raise RuntimeError(f"No completed nnU-Net EBO trials found in {args.results_base}")

    best_trial = min(trials, key=trial_key)
    payload = {
        "selection_rule": "min_val_loss_then_min_val_energy_fpr95",
        "results_base": str(args.results_base),
        "best_trial": best_trial,
        "num_completed_trials": len(trials),
        "trials": trials,
    }

    output_json = args.output_json or (args.results_base / "nnunet_grid_search_best.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    print(json.dumps(payload["best_trial"], indent=2, allow_nan=True))
    print(f"Saved selection to: {output_json}")


if __name__ == "__main__":
    main()
