"""Select the best trials of a (non-nnU-Net) EBO grid search.

Two selections are produced, so that testing can compare a purely segmentation
oriented pick against an OOD oriented one:

  * ``dice``          - the trial with the best validation Dice.
  * ``dice_top_fpr95``- among the top ``--top-frac`` trials by validation Dice,
                        the one with the lowest energy FPR95.

Both selections rank trials using the metrics of the epoch whose checkpoint was
actually written to disk by train_unet.py: ``best_<loss>.pt`` is the minimum
val-loss epoch, ``last_<loss>.pt`` is the final epoch. Ranking on any other
epoch would name a model that cannot be loaded back at test time.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

DICE_CRITERION = 'dice'
TOP_DICE_FPR95_CRITERION = 'dice_top_fpr95'
CRITERIA = (DICE_CRITERION, TOP_DICE_FPR95_CRITERION)


def selection_filename(criterion: str) -> str:
    return f'grid_search_best_{criterion}.json'


def checkpoint_filename(loss: str, selection_mode: str) -> str:
    prefix = 'last' if selection_mode == 'last' else 'best'
    return f'{prefix}_{loss}.pt'


def _as_float(value: Any, default: float) -> float:
    """Metrics can be None (FPR95 needs at least one error pixel) or NaN."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _jsonable(value: float) -> Optional[float]:
    """The +/-inf sentinels used for ranking are not valid JSON."""
    return value if math.isfinite(value) else None


def _jsonable_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    return {**trial, 'dice': _jsonable(trial['dice']), 'fpr95': _jsonable(trial['fpr95'])}


def selection_epoch(history: List[Dict[str, Any]], selection_mode: str) -> Dict[str, Any]:
    """Return the history entry matching the checkpoint kept on disk."""
    if not history:
        raise ValueError('Historique vide.')
    if selection_mode == 'last':
        return history[-1]
    return min(history, key=lambda item: item['val']['loss'])


def collect_trials(
    results_dir: Path,
    loss: str,
    selection_mode: str = 'best',
    expected_epochs: Optional[int] = None,
    require_checkpoint: bool = True,
) -> List[Dict[str, Any]]:
    """Scan ``results_dir`` for finished trials and read their checkpoint metrics.

    ``expected_epochs`` filters out trials that are still running or that
    crashed: train_unet.py rewrites the history file after every epoch, so a
    partial history is perfectly readable and would otherwise be ranked against
    fully trained ones.
    """
    trials: List[Dict[str, Any]] = []
    checkpoint_name = checkpoint_filename(loss, selection_mode)

    for history_path in sorted(results_dir.glob(f'trial_*/history_{loss}.json')):
        trial_dir = history_path.parent
        try:
            history = json.loads(history_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            # A trial writing its history while we read it.
            continue
        if not history:
            continue
        if expected_epochs is not None and len(history) < expected_epochs:
            continue

        checkpoint_path = trial_dir / checkpoint_name
        if require_checkpoint and not checkpoint_path.is_file():
            continue

        epoch_entry = selection_epoch(history, selection_mode)
        params = None
        config_path = trial_dir / 'grid_trial.json'
        if config_path.is_file():
            try:
                params = json.loads(config_path.read_text(encoding='utf-8')).get('params')
            except json.JSONDecodeError:
                params = None

        trials.append({
            'trial_name': trial_dir.name,
            'trial_dir': str(trial_dir),
            'checkpoint': str(checkpoint_path),
            'epoch': epoch_entry.get('epoch'),
            'num_epochs': len(history),
            'params': params,
            'val_metrics': epoch_entry['val'],
            'dice': _as_float(epoch_entry['val'].get('dice'), -math.inf),
            'fpr95': _as_float(epoch_entry['val'].get('fpr95'), math.inf),
        })

    return trials


def select_best_dice(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(trials, key=lambda trial: (trial['dice'], -trial['fpr95']))


def top_dice_pool(trials: List[Dict[str, Any]], top_frac: float) -> List[Dict[str, Any]]:
    """The best ``top_frac`` trials by Dice, ties at the cutoff included."""
    if not 0 < top_frac <= 1:
        raise ValueError('--top-frac doit etre dans ]0, 1].')
    ranked = sorted(trials, key=lambda trial: trial['dice'], reverse=True)
    keep = max(1, math.ceil(top_frac * len(ranked)))
    cutoff_dice = ranked[keep - 1]['dice']
    return [trial for trial in ranked if trial['dice'] >= cutoff_dice]


def select_best_fpr95_in_top_dice(trials: List[Dict[str, Any]], top_frac: float) -> Dict[str, Any]:
    pool = top_dice_pool(trials, top_frac)
    return min(pool, key=lambda trial: (trial['fpr95'], -trial['dice']))


def build_payload(
    criterion: str,
    best_trial: Dict[str, Any],
    trials: List[Dict[str, Any]],
    loss: str,
    selection_mode: str,
    top_frac: Optional[float],
    results_dir: Path,
) -> Dict[str, Any]:
    if criterion == DICE_CRITERION:
        metric = 'dice'
        selection_rule = 'max_val_dice_at_checkpoint_epoch'
        value = best_trial['dice']
        pool = None
    else:
        metric = 'fpr95'
        selection_rule = f'min_val_fpr95_among_top_{top_frac:g}_val_dice_at_checkpoint_epoch'
        value = best_trial['fpr95']
        pool = [
            {
                'trial_name': trial['trial_name'],
                'dice': _jsonable(trial['dice']),
                'fpr95': _jsonable(trial['fpr95']),
            }
            for trial in top_dice_pool(trials, top_frac)
        ]

    payload = {
        'criterion': criterion,
        'metric': metric,
        'selection_rule': selection_rule,
        'selection_mode': selection_mode,
        'checkpoint_epoch_rule': (
            'last_epoch' if selection_mode == 'last' else 'min_val_loss_epoch'
        ),
        'loss': loss,
        'results_dir': str(results_dir),
        'top_frac': top_frac,
        'num_trials': len(trials),
        'trial_name': best_trial['trial_name'],
        'trial_dir': best_trial['trial_dir'],
        'checkpoint': best_trial['checkpoint'],
        'epoch': best_trial['epoch'],
        'best_value': _jsonable(value),
        'best_params': best_trial['params'],
        'val_metrics': best_trial['val_metrics'],
    }
    if pool is not None:
        payload['top_dice_pool'] = pool
    return payload


def write_selections(
    results_dir: Path,
    loss: str,
    selection_mode: str = 'best',
    top_frac: float = 0.2,
    expected_epochs: Optional[int] = None,
    require_checkpoint: bool = True,
) -> Dict[str, Path]:
    """Write one JSON per criterion and return their paths."""
    trials = collect_trials(
        results_dir,
        loss,
        selection_mode=selection_mode,
        expected_epochs=expected_epochs,
        require_checkpoint=require_checkpoint,
    )
    if not trials:
        raise RuntimeError(
            f'Aucun essai exploitable dans {results_dir} '
            f'(history_{loss}.json + {checkpoint_filename(loss, selection_mode)} attendus).'
        )

    selections = {
        DICE_CRITERION: (select_best_dice(trials), None),
        TOP_DICE_FPR95_CRITERION: (select_best_fpr95_in_top_dice(trials, top_frac), top_frac),
    }

    written: Dict[str, Path] = {}
    for criterion, (best_trial, criterion_top_frac) in selections.items():
        payload = build_payload(
            criterion, best_trial, trials, loss, selection_mode, criterion_top_frac, results_dir
        )
        output_path = results_dir / selection_filename(criterion)
        output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        written[criterion] = output_path

    summary_path = results_dir / 'grid_search_selection_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'results_dir': str(results_dir),
                'loss': loss,
                'selection_mode': selection_mode,
                'top_frac': top_frac,
                'num_trials': len(trials),
                'selected': {
                    criterion: best_trial['trial_name']
                    for criterion, (best_trial, _) in selections.items()
                },
                'trials': [
                    _jsonable_trial(trial)
                    for trial in sorted(trials, key=lambda trial: trial['dice'], reverse=True)
                ],
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    written['summary'] = summary_path
    return written


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Selectionne les 2 meilleurs essais d une grid search EBO (best Dice, '
                    'puis best FPR95 parmi les meilleurs Dice).'
    )
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Dossier de la grid search contenant les trial_*/.')
    parser.add_argument('--loss', type=str, required=True,
                        help='Nom normalise de la loss (suffixe de history_<loss>.json).')
    parser.add_argument('--selection-mode', choices=['best', 'last'], default='best',
                        help='best: epoque de best_<loss>.pt (min val loss). last: last_<loss>.pt.')
    parser.add_argument('--top-frac', type=float, default=0.2,
                        help='Fraction des meilleurs Dice gardee avant de minimiser FPR95.')
    parser.add_argument('--expected-epochs', type=int, default=None,
                        help='Ignore les essais dont l historique compte moins d epoques (essais en cours ou plantes).')
    parser.add_argument('--allow-missing-checkpoint', action='store_true',
                        help='Classe aussi les essais dont le checkpoint a ete supprime.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    from losses import normalize_loss_name

    written = write_selections(
        args.results_dir,
        normalize_loss_name(args.loss),
        selection_mode=args.selection_mode,
        top_frac=args.top_frac,
        expected_epochs=args.expected_epochs,
        require_checkpoint=not args.allow_missing_checkpoint,
    )

    for criterion in CRITERIA:
        payload = json.loads(written[criterion].read_text(encoding='utf-8'))
        print(f'[{criterion}] {payload["trial_name"]} '
              f'dice={payload["val_metrics"].get("dice")} '
              f'fpr95={payload["val_metrics"].get("fpr95")}')
        print(f'  checkpoint: {payload["checkpoint"]}')
    print(f'Selections ecrites dans: {args.results_dir}')


if __name__ == '__main__':
    main()
