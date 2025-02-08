import numpy as np
import pickle
import os
import argparse

from utils.utils import compute_robustness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--safety_threshold', type=float, default=0.3, help='Min distance from walls.')
    args = parser.parse_args()
    print(args)

    np.random.seed(0)

    val_predictions = f'models/{args.dataset_seed}/eval/id/val_predictions.pkl'
    with open(val_predictions, 'rb') as f:
        val_predictions = pickle.load(f)
    n_episodes = len(val_predictions.keys())
    
    # to construct independent set, pull one random window from each simulation episode
    cal_set = []
    for epi in val_predictions.keys():
        hist = val_predictions[epi]['X']
        horz = val_predictions[epi]['Y']
        horz_pred = val_predictions[epi]['Y_pred']

        assert len(hist) == len(horz) == len(horz_pred)
        n_steps = len(hist)

        idx = np.random.randint(0, n_steps)
        rho = compute_robustness(trajectories = horz[idx,:,:], safety_threshold = args.safety_threshold)
        rho_hat = compute_robustness(trajectories = horz_pred[idx,:,:], safety_threshold = args.safety_threshold)
        cal_set.append(rho_hat - rho)
        
    print(f'calibration set size: {len(cal_set)}')
        
    file_pth = f'calibration_sets/calibration_set_{args.dataset_seed}.pkl'
    with open(file_pth, 'wb') as f:
        pickle.dump(cal_set, f)
