"""Select the best trials of an nnU-Net EBO grid search.

Same metrics and same selection rules as select_grid_best.py (the U-Net /
TransUNet grid search), only the on-disk layout differs: nnU-Net trials keep
their per-epoch metrics in ``trial_*/**/fold_*/ebo_validation_history.json``
written by nnunet_trainer.py, and their checkpoints are ``checkpoint_best.pth``
(min val loss) and ``checkpoint_final.pth`` (last epoch).

Three selections are produced:

  * ``dice``          - the trial with the best validation Dice.
  * ``dice_top_fpr95``- among the top ``--top-frac`` trials by validation Dice,
                        the one with the lowest energy FPR95.
  * ``fpr95``         - the trial with the lowest energy FPR95, over every
                        config, with no Dice constraint.

Trials are ranked on the metrics of the epoch whose checkpoint is actually on
disk: ``--selection-mode best`` uses the min val-loss epoch, ``last`` the final
one. The Dice is nnU-Net's mean foreground pseudo-Dice, the FPR95 is the energy
FPR95 computed by the trainer over sampled validation pixels.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from select_grid_best import (
    CRITERIA,
    DICE_CRITERION,
    _as_float,
    selection_epoch,
    write_selections_from_trials,
)

BEST_CHECKPOINT = 'checkpoint_best.pth'
LAST_CHECKPOINT = 'checkpoint_final.pth'
HISTORY_NAME = 'ebo_validation_history.json'
SUMMARY_NAME = 'nnunet_grid_search_selection_summary.json'
# Kept for the older consumers (inference_nnunet.sh, compute_d_matrix_nnunet.sh,
# inference_nnunet.py --best-json) that expect a single winner.
LEGACY_BEST_JSON = 'nnunet_grid_search_best.json'


def selection_filename(criterion: str) -> str:
    return f'nnunet_grid_search_best_{criterion}.json'


def checkpoint_filename(selection_mode: str) -> str:
    return LAST_CHECKPOINT if selection_mode == 'last' else BEST_CHECKPOINT


def _legacy_epoch_metrics(
    checkpoint_path: Path,
    history: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Rebuild the checkpoint-epoch metrics of a trial trained before the trainer
    logged dice/fpr95 per epoch.

    Nothing has to be retrained: nnU-Net stores its whole logger inside the
    checkpoint, truncated at the epoch that was saved, so the last entry of
    val_losses / mean_fg_dice is exactly the checkpoint epoch. The FPR95 of that
    epoch is in the old history, because the old trainer computed it precisely
    on the epochs where the checkpoint was written.
    """
    import torch  # local: the selection must stay importable without torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logging = checkpoint.get("logging") or {}
    val_losses = logging.get("val_losses") or []
    dices = logging.get("mean_fg_dice") or []
    if not val_losses or not dices:
        return None

    epoch = len(val_losses) - 1
    by_epoch = {entry.get("epoch"): entry for entry in history}
    entry = by_epoch.get(epoch, {})
    fpr95 = entry.get("val_energy_fpr95", math.nan)
    if not isinstance(fpr95, (int, float)) or math.isnan(float(fpr95)):
        # The summary keeps the FPR95 of the selected checkpoint epoch.
        fpr95 = summary.get("best_val_energy_fpr95", math.nan)

    return {
        "epoch": epoch,
        "loss": float(val_losses[-1]),
        "dice": float(dices[-1]),
        "fpr95": fpr95,
        "source": "legacy: checkpoint logging + ebo_validation_history",
    }


def _trial_params(trial_root: Path) -> Optional[Dict[str, Any]]:
    """Hyperparameters written next to the trial by nnunet_grid_search.sh."""
    config_path = trial_root / 'grid_trial.json'
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding='utf-8')).get('params')
    except json.JSONDecodeError:
        return None


def collect_trials(
    results_base: Path,
    selection_mode: str = 'best',
    fold: Optional[int] = None,
    expected_epochs: Optional[int] = None,
    require_checkpoint: bool = True,
) -> List[Dict[str, Any]]:
    """Scan ``results_base`` for finished trials and read their checkpoint metrics."""
    trials: List[Dict[str, Any]] = []
    legacy_trials: List[str] = []
    checkpoint_name = checkpoint_filename(selection_mode)

    for history_path in sorted(results_base.glob(f'trial_*/**/{HISTORY_NAME}')):
        fold_dir = history_path.parent
        if fold is not None and fold_dir.name != f'fold_{fold}':
            continue

        try:
            history = json.loads(history_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            # A trial rewriting its history while we read it.
            continue
        if not history:
            continue
        if expected_epochs is not None and len(history) < expected_epochs:
            continue

        checkpoint_path = fold_dir / checkpoint_name
        if require_checkpoint and not checkpoint_path.is_file():
            continue

        # Histories written before the trainer logged dice/fpr95 every epoch
        # have no 'val' block; their checkpoint-epoch metrics are rebuilt from
        # the checkpoint itself rather than dropping (or retraining) the trial.
        if all('val' in entry for entry in history):
            epoch_entry = selection_epoch(history, selection_mode)
            metrics = {'epoch': epoch_entry.get('epoch'), **epoch_entry['val']}
        else:
            if not checkpoint_path.is_file():
                continue
            summary_path = fold_dir / 'ebo_selection_summary.json'
            summary = (
                json.loads(summary_path.read_text(encoding='utf-8'))
                if summary_path.is_file() else {}
            )
            metrics = _legacy_epoch_metrics(checkpoint_path, history, summary)
            if metrics is None:
                continue
            legacy_trials.append(fold_dir.parent.parent.name)

        trial_root = results_base / history_path.relative_to(results_base).parts[0]

        trials.append({
            'trial_name': trial_root.name,
            'trial_dir': str(fold_dir),
            'trial_root': str(trial_root),
            'fold': fold_dir.name,
            'checkpoint': str(checkpoint_path),
            'epoch': metrics.get('epoch'),
            'num_epochs': len(history),
            'params': _trial_params(trial_root),
            'val_metrics': metrics,
            'loss': _as_float(metrics.get('loss'), math.inf),
            'dice': _as_float(metrics.get('dice'), -math.inf),
            'fpr95': _as_float(metrics.get('fpr95'), math.inf),
        })

    if legacy_trials:
        print(f'{len(legacy_trials)} essai(s) au format historique pre-dice: metriques de '
              f"l'epoque du checkpoint relues depuis {checkpoint_name}.")
        missing_fpr95 = [
            trial['trial_name'] for trial in trials
            if not math.isfinite(trial['fpr95'])
        ]
        if missing_fpr95:
            print(f'ATTENTION: {len(missing_fpr95)} essai(s) sans FPR95 a leur epoque de '
                  'checkpoint; les selections fpr95 et dice_top_fpr95 les classent en dernier. '
                  'Avec --selection-mode best le FPR95 est toujours disponible.')

    return trials


def write_legacy_best_json(results_base: Path, selection_path: Path, output_json: Path) -> None:
    """Mirror one criterion into the single-winner JSON the nnU-Net scripts read."""
    payload = json.loads(selection_path.read_text(encoding='utf-8'))
    legacy = {
        **payload,
        'results_base': str(results_base),
        'best_trial': {
            'trial_name': payload['trial_name'],
            'trial_dir': payload['trial_dir'],
            'checkpoint': payload['checkpoint'],
            'epoch': payload['epoch'],
            'val_metrics': payload['val_metrics'],
            'selection_rule': payload['selection_rule'],
        },
        'num_completed_trials': payload['num_trials'],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(legacy, indent=2, allow_nan=True), encoding='utf-8')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Selectionne les 3 meilleurs essais d une grid search nnU-Net EBO '
                    '(best Dice, best FPR95 parmi les meilleurs Dice, best FPR95 tous '
                    'essais confondus), avec les memes regles que select_grid_best.py.'
    )
    parser.add_argument('--results-base', type=Path, required=True,
                        help='Dossier de la grid search contenant les trial_*/.')
    parser.add_argument('--output-json', type=Path, default=None,
                        help=f'Chemin du JSON mono-critere de compatibilite ({LEGACY_BEST_JSON}).')
    parser.add_argument('--fold', type=int, default=None,
                        help='Ne garde que les checkpoints de ce fold.')
    parser.add_argument('--loss', type=str, default='',
                        help='Nom de la loss, recopie dans les JSON de selection.')
    parser.add_argument('--selection-mode', choices=['best', 'last'], default='best',
                        help=f'best: epoque de {BEST_CHECKPOINT} (min val loss). '
                             f'last: {LAST_CHECKPOINT}.')
    parser.add_argument('--top-frac', type=float, default=0.2,
                        help='Fraction des meilleurs Dice gardee avant de minimiser FPR95 '
                             '(selection dice_top_fpr95 uniquement).')
    parser.add_argument('--expected-epochs', type=int, default=None,
                        help='Ignore les essais dont l historique compte moins d epoques '
                             '(essais en cours ou plantes).')
    parser.add_argument('--legacy-criterion', choices=list(CRITERIA) + ['none'],
                        default=DICE_CRITERION,
                        help=f'Critere recopie dans {LEGACY_BEST_JSON} (mono-critere, pour les '
                             'anciens appels). "none" pour n ecrire que les 3 fichiers par critere.')
    parser.add_argument('--allow-missing-checkpoint', action='store_true',
                        help='Classe aussi les essais dont le checkpoint a ete supprime.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    trials = collect_trials(
        args.results_base,
        selection_mode=args.selection_mode,
        fold=args.fold,
        expected_epochs=args.expected_epochs,
        require_checkpoint=not args.allow_missing_checkpoint,
    )
    if not trials:
        raise RuntimeError(
            f'Aucun essai exploitable dans {args.results_base} '
            f'({HISTORY_NAME} + {checkpoint_filename(args.selection_mode)} attendus).'
        )

    written = write_selections_from_trials(
        trials,
        args.results_base,
        args.loss,
        selection_mode=args.selection_mode,
        top_frac=args.top_frac,
        filename_fn=selection_filename,
        summary_filename=SUMMARY_NAME,
        extra_summary={'fold': args.fold},
    )

    def _fmt(value: Any) -> str:
        return 'n/a' if value is None else f'{float(value):.4f}'

    print()
    print(f'{len(trials)} essais classes (mode {args.selection_mode}, top-frac {args.top_frac:g})')
    print('=' * 60)
    for criterion in CRITERIA:
        payload = json.loads(written[criterion].read_text(encoding='utf-8'))
        metrics = payload['val_metrics']
        print(f'[{criterion}] {payload["selection_rule"]}')
        print(f'  essai     : {payload["trial_name"]}')
        print(f'  epoque {payload["epoch"]} : loss={_fmt(metrics.get("loss"))} '
              f'dice={_fmt(metrics.get("dice"))} fpr95={_fmt(metrics.get("fpr95"))}')
        print(f'  checkpoint: {payload["checkpoint"]}')
        print(f'  fichier   : {written[criterion]}')
    print('=' * 60)
    print(f'Recapitulatif de tous les essais: {written["summary"]}')

    if args.legacy_criterion != 'none':
        output_json = args.output_json or (args.results_base / LEGACY_BEST_JSON)
        write_legacy_best_json(args.results_base, written[args.legacy_criterion], output_json)
        print(f'Copie mono-critere pour les anciens appels (= critere '
              f'{args.legacy_criterion}): {output_json}')


if __name__ == '__main__':
    main()
