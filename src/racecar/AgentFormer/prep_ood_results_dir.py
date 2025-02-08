import pickle
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_cfg', default=None)
parser.add_argument('--dataset_seed', default=0)
parser.add_argument('--train_step', choices=['ogtest', 'ft', 'fttest'], default='og', help='Original training step or finetuning step.')
parser.add_argument('--ood_type', choices=[None, 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5'], default=None)
args = parser.parse_args()


if args.train_step in ['ogtest', 'ft']:
    ood_list = ['vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5']
elif args.train_step == 'fttest':
    if args.ood_type is None:
        raise TypeError
    ood_list = [args.ood_type]
else:
    raise NotImplementedError

base_results_dir = os.path.join(f'results_s{args.dataset_seed}', args.train_cfg)
for ood in ood_list:
    # create directory to copy best model to
    base_cp_dir = os.path.join(f'results_s{args.dataset_seed}', f'racetrack_{ood}_{args.train_step}')
    model_cp_dir = os.path.join(base_cp_dir, 'models')
    os.makedirs(model_cp_dir, exist_ok=True)

    # choose best model based on saved training metrics
    metrics_file = os.path.join(base_results_dir, 'validation_metrics.pkl')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
    best_model_pth = os.path.join(base_results_dir, 'models', f'model_{best_epoch:04d}.p')

    # copy best model to created directory
    shutil.copy(best_model_pth, model_cp_dir)

    # copy training metrics to created directory if in testing phase (needed for fp script)
    if args.train_step in ['ogtest', 'fttest']:
        shutil.copy(metrics_file, base_cp_dir)