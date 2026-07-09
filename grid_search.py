import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import math

from losses import normalize_loss_name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Grid search Optuna pour les hyperparametres EBO.')
    parser.add_argument('--dataset', type=str, choices=['acdc', 'brats'], default=None)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument(
        '--pretrained-path', type=Path, default=None,
        help='Pretrained encoder checkpoint forwarded to train_unet.py (e.g. R50+ViT-B_16.npz for transunet).',
    )
    parser.add_argument('--loss', type=str, default='ebo_ce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument(
        '--in-channels', type=int, default=4,
        help='BraTS input channels: 4 (extract_data.py) or 3 (extract_data_brats3.py). Ignored for ACDC.',
    )
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--metric', choices=['loss', 'dice', 'iou', 'pixel_acc','fpr95'], default='dice')
    parser.add_argument('--selection-mode', choices=['best', 'last'], default='best')
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default=None,
        help='Liste de GPU a utiliser, separes par des virgules. Exemple: 0,1,2.',
    )
    parser.add_argument(
        '--models-per-gpu',
        type=int,
        default=3,
        help='Nombre d entrainements lances en parallele sur chaque GPU.',
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=None,
        help='Plafond optionnel du nombre total d entrainements lances en parallele.',
    )
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
    parser.add_argument('--rho-grid', type=float, nargs='+', default=[1.0])
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


def is_bound_aug_lag_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'bound_ebo_aug_lag', 'bound_aug_lag_ebo', 'boundary_aug_lag_ebo'}


def is_bound_aug_log_ebo_loss(loss_name: str) -> bool:
    return loss_name in {'bound_ebo_aug_log', 'bound_aug_log_ebo', 'boundary_aug_log_ebo'}


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

    if is_bound_aug_lag_ebo_loss(args.loss):
        return {
            'lambda_ebo_cen_in': args.lambda_ebo_cen_in_grid,
            'lambda_ebo_out_in': args.lambda_ebo_out_in_grid,
            'lambda_ebo_cen_corr': args.lambda_ebo_cen_corr_grid,
            'lambda_ebo_out_corr': args.lambda_ebo_out_corr_grid,
            'boundary_k': args.boundary_k_grid,
            'margin_correct': args.margin_correct_grid,
            'margin_miss': args.margin_miss_grid,
            'rho': args.rho_grid,
        }

    if is_bound_aug_log_ebo_loss(args.loss):
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
            'rho': args.rho_grid,
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



def resolve_gpu_ids(gpu_ids: str | None) -> List[str]:
    if gpu_ids:
        resolved = [gpu_id.strip() for gpu_id in gpu_ids.split(',') if gpu_id.strip()]
    else:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            resolved = [gpu_id.strip() for gpu_id in visible_devices.split(',') if gpu_id.strip()]
        else:
            resolved = ['0']

    if not resolved:
        raise ValueError('Aucun GPU disponible. Utilise --gpu-ids, par exemple --gpu-ids 0,1,2.')
    return resolved

def main() -> None:
    args = build_argparser().parse_args()
    args.loss = normalize_loss_name(args.loss)
    if (
        not is_standard_ebo_loss(args.loss)
        and not is_bound_ebo_loss(args.loss)
        and not is_log_ebo_loss(args.loss)
        and not is_bound_log_ebo_loss(args.loss)
        and not is_bound_aug_lag_ebo_loss(args.loss)
        and not is_bound_aug_log_ebo_loss(args.loss)
    ):
        raise ValueError('Cette grid search est prevue pour une loss EBO.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_space = build_search_space(args)
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    array_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    def expand_grid(space):
        import itertools
        keys = list(space.keys())
        values = list(space.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    all_configs = list(expand_grid(search_space))
    total = len(all_configs)

    chunk_size = math.ceil(total / array_count)

    start = array_id * chunk_size
    end = min(start + chunk_size, total)

    direction = 'minimize' if args.metric in ['loss','fpr95'] else 'maximize'
    script_path = Path(__file__).resolve().parent / 'train_unet.py'
    results_path = args.output_dir / 'grid_search_results.json'
    best_path = args.output_dir / 'grid_search_best.json'

    def build_trial(trial_number: int, config: Dict[str, Any]) -> Dict[str, Any]:
        margin_correct = config['margin_correct']
        margin_miss = config['margin_miss']

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
            '--in-channels', str(args.in_channels),
            '--num-classes', str(args.num_classes),
            '--seed', str(args.seed),
            '--device', args.device,
            '--margin-correct', str(margin_correct),
            '--margin-miss', str(margin_miss),
        ]

        if args.pretrained_path is not None:
            command.extend(['--pretrained-path', str(args.pretrained_path)])

        if is_standard_ebo_loss(args.loss):
            lambda_ebo_in = config['lambda_ebo_in']
            lambda_ebo_corr = config['lambda_ebo_corr']

            trial_name = (
                f"trial_{trial_number:03d}_"
                f"lin_{sanitize_float(lambda_ebo_in)}_"
                f"lcorr_{sanitize_float(lambda_ebo_corr)}_"
                f"mcorr_{sanitize_float(margin_correct)}_"
                f"mmiss_{sanitize_float(margin_miss)}"
            )

            command.extend([
                '--lambda-ebo-in', str(lambda_ebo_in),
                '--lambda-ebo-corr', str(lambda_ebo_corr),
            ])
        elif is_log_ebo_loss(args.loss):
            lambda_ebo_in = config['lambda_ebo_in']
            lambda_ebo_corr = config['lambda_ebo_corr']
            barrier_t = config['barrier_t']
            barrier_t_growth = config['barrier_t_growth']

            trial_name = (
                f"trial_{trial_number:03d}_"
                f"lin_{sanitize_float(lambda_ebo_in)}_"
                f"lcorr_{sanitize_float(lambda_ebo_corr)}_"
                f"t_{sanitize_float(barrier_t)}_"
                f"tg_{sanitize_float(barrier_t_growth)}_"
                f"mcorr_{sanitize_float(margin_correct)}_"
                f"mmiss_{sanitize_float(margin_miss)}"
            )

            command.extend([
                '--lambda-ebo-in', str(lambda_ebo_in),
                '--lambda-ebo-corr', str(lambda_ebo_corr),
                '--barrier-t', str(barrier_t),
                '--barrier-t-growth', str(barrier_t_growth),
            ])
        elif is_bound_log_ebo_loss(args.loss):
            lambda_ebo_cen_in = config['lambda_ebo_cen_in']
            lambda_ebo_out_in = config['lambda_ebo_out_in']
            lambda_ebo_cen_corr = config['lambda_ebo_cen_corr']
            lambda_ebo_out_corr = config['lambda_ebo_out_corr']
            boundary_k = config['boundary_k']
            barrier_t = config['barrier_t']
            barrier_t_growth = config['barrier_t_growth']

            trial_name = (
                f"trial_{trial_number:03d}_"
                f"cenin_{sanitize_float(lambda_ebo_cen_in)}_"
                f"outin_{sanitize_float(lambda_ebo_out_in)}_"
                f"cencorr_{sanitize_float(lambda_ebo_cen_corr)}_"
                f"outcorr_{sanitize_float(lambda_ebo_out_corr)}_"
                f"bk_{boundary_k}_"
                f"t_{sanitize_float(barrier_t)}_"
                f"tg_{sanitize_float(barrier_t_growth)}_"
                f"mcorr_{sanitize_float(margin_correct)}_"
                f"mmiss_{sanitize_float(margin_miss)}"
            )

            command.extend([
                '--lambda-ebo-cen-in', str(lambda_ebo_cen_in),
                '--lambda-ebo-out-in', str(lambda_ebo_out_in),
                '--lambda-ebo-cen-corr', str(lambda_ebo_cen_corr),
                '--lambda-ebo-out-corr', str(lambda_ebo_out_corr),
                '--boundary-k', str(boundary_k),
                '--barrier-t', str(barrier_t),
                '--barrier-t-growth', str(barrier_t_growth),
            ])
        elif is_bound_aug_lag_ebo_loss(args.loss) or is_bound_aug_log_ebo_loss(args.loss):
            lambda_ebo_cen_in = config['lambda_ebo_cen_in']
            lambda_ebo_out_in = config['lambda_ebo_out_in']
            lambda_ebo_cen_corr = config['lambda_ebo_cen_corr']
            lambda_ebo_out_corr = config['lambda_ebo_out_corr']
            boundary_k = config['boundary_k']
            rho = config['rho']

            barrier_t = config.get('barrier_t')
            barrier_t_growth = config.get('barrier_t_growth')

            log_suffix = (
                f"t_{sanitize_float(barrier_t)}_tg_{sanitize_float(barrier_t_growth)}_"
                if is_bound_aug_log_ebo_loss(args.loss)
                else ""
            )

            trial_name = (
                f"trial_{trial_number:03d}_"
                f"cenin_{sanitize_float(lambda_ebo_cen_in)}_"
                f"outin_{sanitize_float(lambda_ebo_out_in)}_"
                f"cencorr_{sanitize_float(lambda_ebo_cen_corr)}_"
                f"outcorr_{sanitize_float(lambda_ebo_out_corr)}_"
                f"bk_{boundary_k}_"
                f"rho_{sanitize_float(rho)}_"
                f"{log_suffix}"
                f"mcorr_{sanitize_float(margin_correct)}_"
                f"mmiss_{sanitize_float(margin_miss)}"
            )

            command.extend([
                '--lambda-ebo-cen-in', str(lambda_ebo_cen_in),
                '--lambda-ebo-out-in', str(lambda_ebo_out_in),
                '--lambda-ebo-cen-corr', str(lambda_ebo_cen_corr),
                '--lambda-ebo-out-corr', str(lambda_ebo_out_corr),
                '--boundary-k', str(boundary_k),
                '--rho', str(rho),
            ])

            if is_bound_aug_log_ebo_loss(args.loss):
                command.extend([
                    '--barrier-t', str(barrier_t),
                    '--barrier-t-growth', str(barrier_t_growth),
                ])
        else:
            lambda_ebo_cen_in = config['lambda_ebo_cen_in']
            lambda_ebo_out_in = config['lambda_ebo_out_in']
            lambda_ebo_cen_corr = config['lambda_ebo_cen_corr']
            lambda_ebo_out_corr = config['lambda_ebo_out_corr']
            boundary_k = config['boundary_k']

            trial_name = (
                f"trial_{trial_number:03d}_"
                f"cenin_{sanitize_float(lambda_ebo_cen_in)}_"
                f"outin_{sanitize_float(lambda_ebo_out_in)}_"
                f"cencorr_{sanitize_float(lambda_ebo_cen_corr)}_"
                f"outcorr_{sanitize_float(lambda_ebo_out_corr)}_"
                f"bk_{boundary_k}_"
                f"mcorr_{sanitize_float(margin_correct)}_"
                f"mmiss_{sanitize_float(margin_miss)}"
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

        return {
            'number': trial_number,
            'params': config,
            'trial_name': trial_name,
            'trial_dir': trial_dir,
            'command': command,
        }

    total_trials = 1
    for values in search_space.values():
        total_trials *= len(values)

    gpu_ids = resolve_gpu_ids(args.gpu_ids)
    if args.models_per_gpu < 1:
        raise ValueError('--models-per-gpu doit etre >= 1.')
    if args.max_parallel is not None and args.max_parallel < 1:
        raise ValueError('--max-parallel doit etre >= 1.')

    max_parallel = len(gpu_ids) * args.models_per_gpu
    if args.max_parallel is not None:
        max_parallel = min(max_parallel, args.max_parallel)
    if max_parallel < 1:
        raise ValueError('Le parallelisme total doit etre >= 1.')

    trials = [
        build_trial(global_idx, config)
        for global_idx, config in enumerate(all_configs[start:end], start=start)
    ]

    print(f'Nombre total de combinaisons: {total_trials}')
    print(f'Combinaisons pour cette tache: {len(trials)} ({start} a {end - 1})')
    print(
        f'GPUs utilises: {gpu_ids} | '
        f'modeles/GPU: {args.models_per_gpu} | '
        f'parallele total: {max_parallel}'
    )
    print(json.dumps(search_space, indent=2))

    results = []
    for batch_start in range(0, len(trials), max_parallel):
        batch = trials[batch_start:batch_start + max_parallel]
        processes = []

        for slot, trial in enumerate(batch):
            gpu_id = gpu_ids[slot % len(gpu_ids)]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f"[{trial['number'] + 1}/{total_trials}] GPU {gpu_id}: {trial['trial_name']}")
            process = subprocess.Popen(
                trial['command'],
                cwd=script_path.parent,
                env=env,
            )
            processes.append((process, trial, gpu_id))

        failed = []
        for process, trial, gpu_id in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((trial, gpu_id, return_code))
                results.append({
                    'number': trial['number'],
                    'value': None,
                    'params': trial['params'],
                    'state': 'FAIL',
                    'trial_dir': str(trial['trial_dir']),
                    'val_metrics': None,
                    'command': trial['command'],
                    'gpu_id': gpu_id,
                    'return_code': return_code,
                })
                continue

            history_path = trial['trial_dir'] / f'history_{args.loss}.json'
            objective_value, val_metrics = read_objective(history_path, args.metric, args.selection_mode)
            results.append({
                'number': trial['number'],
                'value': objective_value,
                'params': trial['params'],
                'state': 'COMPLETE',
                'trial_dir': str(trial['trial_dir']),
                'val_metrics': val_metrics,
                'command': trial['command'],
                'gpu_id': gpu_id,
            })

        results_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        if failed:
            details = ', '.join(
                f"{trial['trial_name']} sur GPU {gpu_id} (code {return_code})"
                for trial, gpu_id, return_code in failed
            )
            raise subprocess.CalledProcessError(failed[0][2], failed[0][0]['command'], details)

    completed_trials = [trial for trial in results if trial['value'] is not None]
    if not completed_trials:
        raise RuntimeError('Aucun essai complete dans la grid search.')

    best_trial = (
        min(completed_trials, key=lambda trial: trial['value'])
        if direction == 'minimize'
        else max(completed_trials, key=lambda trial: trial['value'])
    )
    best_payload = {
        'metric': args.metric,
        'selection_mode': args.selection_mode,
        'direction': direction,
        'best_value': best_trial['value'],
        'best_params': best_trial['params'],
        'trial_dir': best_trial['trial_dir'],
        'val_metrics': best_trial['val_metrics'],
    }
    if args.metric == 'fpr95' and args.selection_mode == 'best':
        best_payload['selection_rule'] = 'best_val_loss_per_config_then_min_fpr95'
    best_path.write_text(json.dumps(best_payload, indent=2), encoding='utf-8')

    print('Meilleur essai:')
    print(json.dumps(best_payload, indent=2))
    print(f'Resultats sauvegardes dans: {results_path}')


if __name__ == '__main__':
    main()
