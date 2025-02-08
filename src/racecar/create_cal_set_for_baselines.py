import numpy as np
import pickle
import os
import argparse

from utils.utils import *
from AgentFormer.utils.utils import find_unique_common_from_lists, load_list_from_folder, load_txt_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--safety_threshold', type=float, default=5.4, help='Min distance from other vehicles.')
    args = parser.parse_args()
    print(args)

    np.random.seed(0)
    dataset_base = f'AgentFormer/datasets/racetrack/id_s{args.dataset_seed}'

    # get predictions path
    model_base = f'AgentFormer/results_s{args.dataset_seed}/racetrack_agentformer_pre/'
    metrics_file = os.path.join(model_base, 'validation_metrics.pkl')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
    val_predictions_base = os.path.join(model_base, f'results/epoch_{best_epoch:04d}/val/recon')

    # prepare to load data
    _, seq_val, _ = get_racetrack_split_for_fp(dataset_base)

    # get dict of ego vehicle IDs
    with open(os.path.join(dataset_base, f'ego_ids_val.pkl'), 'rb') as f:
        ego_ids_dict = pickle.load(f)

    cal_set = []
    for seq_name in seq_val:
        # load gt raw data and crash labels from datasets folder
        gt_raw, gt_raw_all, crashes = load_groundtruth_data(f'{dataset_base}', seq_name)
        data_filelist, _ = load_list_from_folder(os.path.join(val_predictions_base, seq_name))

        # get random window
        n_steps = len(data_filelist)
        idx = np.random.randint(0, n_steps)
        data_file = data_filelist[idx]

        # load reconstructed data from AgentFormer, and process reconstructed and groundtruth data
        ego_id = float(ego_ids_dict[int(seq_name.split('_')[1])])
        horz_gt, horz_pred = process_reconstructed_and_gt_data(data_file, gt_raw, ego_id)

        rho = compute_robustness(trajectories = horz_gt, safety_threshold = args.safety_threshold)
        rho_hat = compute_robustness(trajectories = horz_pred, safety_threshold = args.safety_threshold)
        cal_set.append(rho_hat - rho)

    print(f'calibration set size: {len(cal_set)}')
        
    with open(f'calibration_sets/calibration_set_{args.dataset_seed}.pkl', 'wb') as f:
        pickle.dump(cal_set, f)
