import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import optuna

from losses import normalize_loss_name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Grid search Optuna pour les hyperparametres EBO.')
    parser.add_argument('--dataset', type=str, choices=['acdc', 'brats'], default=None)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--loss', type=str, default='ebo_ce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--metric', choices=['loss', 'dice', 'iou', 'pixel_acc','fpr95'], default='dice')
    parser.add_argument('--selection-mode', choices=['best', 'last'], default='best')
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument('--lambda-ebo-in-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--lambda-ebo-corr-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--lambda-ebo-cen-in-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--lambda-ebo-out-in-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--lambda-ebo-cen-corr-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--lambda-ebo-out-corr-grid', type=float, nargs='+', default=[0.1])
    parser.add_argument('--boundary-k-grid', type=int, nargs='+', default=[1])
    parser.add_argument('--margin-correct-grid', type=float, nargs='+', default=[-35.0])
    parser.add_argument('--margin-miss-grid', type=float, nargs='+', default=[-5.0])
    parser.add_argument('--barrier-t-grid', type=float, nargs='+', default=[1.0])
    parser.add_argument(
        '--barrier-t-growth-grid',
        '--barrier-t-growth',
        dest='barrier_t_growth_grid',
        type=float,
        nargs='+',
        default=[1.1],
    )
    return parser


def sanitize_float(value: float) -> str:
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def is_standard_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'ebo_ce', 'ebo_cross_entropy'}


def is_bound_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'bound_ebo_ce', 'bound_ebo_cross_entropy', 'boundary_ebo_ce'}


def is_log_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'log_ebo'}


def is_bound_log_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'bound_log_ebo', 'bound_ebo_log_barrier', 'boundary_log_ebo'}


def build_search_space(args: argparse.Namespace) -> Dict[str, List[float]]:
    if is_standard_ebo_loss(args.loss):
        return {
            'lambda_ebo_in': args.lambda_ebo_in_grid,
            'lambda_ebo_corr': args.lambda_ebo_corr_grid,
            'margin_correct': args.margin_correct_grid,
            'margin_miss': args.margin_miss_grid,
        }

    if is_bound_ebo_loss(args.loss):
        return {
            'lambda_ebo_cen_in': args.lambda_ebo_cen_in_grid,
            'lambda_ebo_out_in': args.lambda_ebo_out_in_grid,
            'lambda_ebo_cen_corr': args.lambda_ebo_cen_corr_grid,
            'lambda_ebo_out_corr': args.lambda_ebo_out_corr_grid,
            'boundary_k': args.boundary_k_grid,
            'margin_correct': args.margin_correct_grid,
            'margin_miss': args.margin_miss_grid,
        }

    if is_log_ebo_loss(args.loss):
        return {
            'lambda_ebo_in': args.lambda_ebo_in_grid,
            'lambda_ebo_corr': args.lambda_ebo_corr_grid,
            'margin_correct': args.margin_correct_grid,
            'margin_miss': args.margin_miss_grid,
            'barrier_t': args.barrier_t_grid,
            'barrier_t_growth': args.barrier_t_growth_grid,
        }

    if is_bound_log_ebo_loss(args.loss):
        return {
            'lambda_ebo_cen_in': args.lambda_ebo_cen_in_grid,
            'lambda_ebo_out_in': args.lambda_ebo_out_in_grid,
            'lambda_ebo_cen_corr': args.lambda_ebo_cen_corr_grid,
            'lambda_ebo_out_corr': args.lambda_ebo_out_corr_grid,
            'boundary_k': args.boundary_k_grid,
            'margin_correct': args.margin_correct_grid,
            'margin_miss': args.margin_miss_grid,
            'barrier_t': args.barrier_t_grid,
            'barrier_t_growth': args.barrier_t_growth_grid,
        }

    raise ValueError(f'Cette grid search ne prend pas en charge la loss {args.loss}.')


def read_objective(history_path: Path, metric: str, selection_mode: str) -> Tuple[float, Dict[str, float]]:
    history = json.loads(history_path.read_text(encoding='utf-8'))
    if not history:
        raise ValueError(f'Aucun historique trouve dans {history_path}')

    if selection_mode == 'last':
        chosen_epoch = history[-1]
    else:
        if metric == 'fpr95':
            # For FPR95 model selection, first keep the best-loss checkpoint per
            # configuration, then compare configurations using that checkpoint's FPR95.
            chosen_epoch = min(history, key=lambda item: item['val']['loss'])
        elif metric == 'loss':
            chosen_epoch = min(history, key=lambda item: item['val'][metric])
        else:
            chosen_epoch = max(history, key=lambda item: item['val'][metric])

    return chosen_epoch['val'][metric], chosen_epoch['val']


def main() -> None:
    args = build_argparser().parse_args()
    args.loss = normalize_loss_name(args.loss)
    if (
        not is_standard_ebo_loss(args.loss)
        and not is_bound_ebo_loss(args.loss)
        and not is_log_ebo_loss(args.loss)
        and not is_bound_log_ebo_loss(args.loss)
    ):
        raise ValueError('Cette grid search est prevue pour une loss EBO.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_space = build_search_space(args)
    sampler = optuna.samplers.GridSampler(search_space)
    direction = 'minimize' if args.metric in ['loss','fpr95'] else 'maximize'
    study = optuna.create_study(direction=direction, sampler=sampler)
    script_path = Path(__file__).resolve().parent / 'train_unet.py'
    results_path = args.output_dir / 'grid_search_results.json'
    best_path = args.output_dir / 'grid_search_best.json'

    def objective(trial: optuna.Trial) -> float:
        margin_correct = trial.suggest_categorical('margin_correct', search_space['margin_correct'])
        margin_miss = trial.suggest_categorical('margin_miss', search_space['margin_miss'])

        command = [
            args.python_bin,
            str(script_path),
            '--dataset', args.dataset,
            '--dataset-root', str(args.dataset_root),
            '--output-dir', str(args.output_dir),
            '--model', args.model,
            '--loss', args.loss,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--num-workers', str(args.num_workers),
            '--image-size', str(args.image_size),
            '--num-classes', str(args.num_classes),
            '--seed', str(args.seed),
            '--device', args.device,
            '--margin-correct', str(margin_correct),
            '--margin-miss', str(margin_miss),
        ]

        if is_standard_ebo_loss(args.loss):
            lambda_ebo_in = trial.suggest_categorical('lambda_ebo_in', search_space['lambda_ebo_in'])
            lambda_ebo_corr = trial.suggest_categorical('lambda_ebo_corr', search_space['lambda_ebo_corr'])
            trial_name = (
                f'trial_{trial.number:03d}_'
                f'lin_{sanitize_float(lambda_ebo_in)}_'
                f'lcorr_{sanitize_float(lambda_ebo_corr)}_'
                f'mcorr_{sanitize_float(margin_correct)}_'
                f'mmiss_{sanitize_float(margin_miss)}'
            )
            command.extend([
                '--lambda-ebo-in', str(lambda_ebo_in),
                '--lambda-ebo-corr', str(lambda_ebo_corr),
            ])
        elif is_log_ebo_loss(args.loss):
            lambda_ebo_in = trial.suggest_categorical('lambda_ebo_in', search_space['lambda_ebo_in'])
            lambda_ebo_corr = trial.suggest_categorical('lambda_ebo_corr', search_space['lambda_ebo_corr'])
            barrier_t = trial.suggest_categorical('barrier_t', search_space['barrier_t'])
            barrier_t_growth = trial.suggest_categorical('barrier_t_growth', search_space['barrier_t_growth'])
            trial_name = (
                f'trial_{trial.number:03d}_'
                f'lin_{sanitize_float(lambda_ebo_in)}_'
                f'lcorr_{sanitize_float(lambda_ebo_corr)}_'
                f't_{sanitize_float(barrier_t)}_'
                f'tg_{sanitize_float(barrier_t_growth)}_'
                f'mcorr_{sanitize_float(margin_correct)}_'
                f'mmiss_{sanitize_float(margin_miss)}'
            )
            command.extend([
                '--lambda-ebo-in', str(lambda_ebo_in),
                '--lambda-ebo-corr', str(lambda_ebo_corr),
                '--barrier-t', str(barrier_t),
                '--barrier-t-growth', str(barrier_t_growth),
            ])
        elif is_bound_log_ebo_loss(args.loss):
            lambda_ebo_cen_in = trial.suggest_categorical('lambda_ebo_cen_in', search_space['lambda_ebo_cen_in'])
            lambda_ebo_out_in = trial.suggest_categorical('lambda_ebo_out_in', search_space['lambda_ebo_out_in'])
            lambda_ebo_cen_corr = trial.suggest_categorical('lambda_ebo_cen_corr', search_space['lambda_ebo_cen_corr'])
            lambda_ebo_out_corr = trial.suggest_categorical('lambda_ebo_out_corr', search_space['lambda_ebo_out_corr'])
            boundary_k = trial.suggest_categorical('boundary_k', search_space['boundary_k'])
            barrier_t = trial.suggest_categorical('barrier_t', search_space['barrier_t'])
            barrier_t_growth = trial.suggest_categorical('barrier_t_growth', search_space['barrier_t_growth'])
            trial_name = (
                f'trial_{trial.number:03d}_'
                f'cenin_{sanitize_float(lambda_ebo_cen_in)}_'
                f'outin_{sanitize_float(lambda_ebo_out_in)}_'
                f'cencorr_{sanitize_float(lambda_ebo_cen_corr)}_'
                f'outcorr_{sanitize_float(lambda_ebo_out_corr)}_'
                f'bk_{boundary_k}_'
                f't_{sanitize_float(barrier_t)}_'
                f'tg_{sanitize_float(barrier_t_growth)}_'
                f'mcorr_{sanitize_float(margin_correct)}_'
                f'mmiss_{sanitize_float(margin_miss)}'
            )
            command.extend([
                '--lambda-ebo-cen-in', str(lambda_ebo_cen_in),
                '--lambda-ebo-out-in', str(lambda_ebo_out_in),
                '--lambda-ebo-cen-corr', str(lambda_ebo_cen_corr),
                '--lambda-ebo-out-corr', str(lambda_ebo_out_corr),
                '--boundary-k', str(boundary_k),
                '--barrier-t', str(barrier_t),
                '--barrier-t-growth', str(barrier_t_growth)
            ])
        else:
            lambda_ebo_cen_in = trial.suggest_categorical('lambda_ebo_cen_in', search_space['lambda_ebo_cen_in'])
            lambda_ebo_out_in = trial.suggest_categorical('lambda_ebo_out_in', search_space['lambda_ebo_out_in'])
            lambda_ebo_cen_corr = trial.suggest_categorical('lambda_ebo_cen_corr', search_space['lambda_ebo_cen_corr'])
            lambda_ebo_out_corr = trial.suggest_categorical('lambda_ebo_out_corr', search_space['lambda_ebo_out_corr'])
            boundary_k = trial.suggest_categorical('boundary_k', search_space['boundary_k'])
            trial_name = (
                f'trial_{trial.number:03d}_'
                f'cenin_{sanitize_float(lambda_ebo_cen_in)}_'
                f'outin_{sanitize_float(lambda_ebo_out_in)}_'
                f'cencorr_{sanitize_float(lambda_ebo_cen_corr)}_'
                f'outcorr_{sanitize_float(lambda_ebo_out_corr)}_'
                f'bk_{boundary_k}_'
                f'mcorr_{sanitize_float(margin_correct)}_'
                f'mmiss_{sanitize_float(margin_miss)}'
            )
            command.extend([
                '--lambda-ebo-cen-in', str(lambda_ebo_cen_in),
                '--lambda-ebo-out-in', str(lambda_ebo_out_in),
                '--lambda-ebo-cen-corr', str(lambda_ebo_cen_corr),
                '--lambda-ebo-out-corr', str(lambda_ebo_out_corr),
                '--boundary-k', str(boundary_k),
            ])

        trial_dir = args.output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_dir_idx = command.index('--output-dir') + 1
        command[output_dir_idx] = str(trial_dir)

        command = [arg for arg in command if arg is not None]

        print(f'[{trial.number + 1}/{total_trials}] Lancement: {trial_name}')
        subprocess.run(command, check=True, cwd=script_path.parent)

        history_path = trial_dir / f'history_{args.loss}.json'
        objective_value, val_metrics = read_objective(history_path, args.metric, args.selection_mode)

        trial.set_user_attr('trial_dir', str(trial_dir))
        trial.set_user_attr('val_metrics', val_metrics)
        trial.set_user_attr('command', command)

        return objective_value

    total_trials = 1
    for values in search_space.values():
        total_trials *= len(values)

    print(f'Nombre total de combinaisons: {total_trials}')
    print(json.dumps(search_space, indent=2))

    study.optimize(objective, n_trials=total_trials)

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        raise RuntimeError('Aucun essai complete dans la grid search.')

    results = []
    for trial in study.trials:
        result = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'trial_dir': trial.user_attrs.get('trial_dir'),
            'val_metrics': trial.user_attrs.get('val_metrics'),
            'command': trial.user_attrs.get('command'),
        }
        results.append(result)

    results_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    best_trial = study.best_trial
    best_payload = {
        'metric': args.metric,
        'selection_mode': args.selection_mode,
        'direction': direction,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'trial_dir': best_trial.user_attrs.get('trial_dir'),
        'val_metrics': best_trial.user_attrs.get('val_metrics'),
    }
    if args.metric == 'fpr95' and args.selection_mode == 'best':
        best_payload['selection_rule'] = 'best_val_loss_per_config_then_min_fpr95'
    best_path.write_text(json.dumps(best_payload, indent=2), encoding='utf-8')

    print('Meilleur essai:')
    print(json.dumps(best_payload, indent=2))
    print(f'Resultats sauvegardes dans: {results_path}')


if __name__ == '__main__':
    main()
