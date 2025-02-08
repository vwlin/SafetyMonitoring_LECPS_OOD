# select threshold for triggering incremental learning based on ID data
# trigger if NCS = rho_hat - rho > tau (we tend to predict safer conditions than is really the case)

import pickle
import glob
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--quantile', type=float, default=0.8)
parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
parser.add_argument('--f0_predictions_base', type=str, default='models', help='Base path to non-finetuned model predictions.')
parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location to save selected taus.')
args = parser.parse_args()

def calc_threshold(windows_root, quantile=0.9):
    files = glob.glob(f'{windows_root}/*ood_windows.pkl')

    ncs = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)

        steps = list(data.keys())

        for step in steps:
            ncs.append(np.abs(data[step]['ncs']))

    tau = np.quantile(ncs, quantile)
    return tau

def calc_windows_above_threshold(windows_root, tau):
    files = glob.glob(f'{windows_root}/*ood_windows.pkl')
    
    ncs = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        
        steps = list(data.keys())

        for step in steps:
            ncs.append(np.abs(data[step]['ncs']))

    n_above = np.sum(ncs > tau)
    return n_above, len(ncs)

if __name__ == '__main__':
    print(args)

    # load or create new tau dictionary
    if os.path.exists(args.tau_save_pth):
        with open(args.tau_save_pth, 'rb') as f:
            saved_taus = pickle.load(f)
    else:
        saved_taus = {}

    # select tau
    tau = calc_threshold(os.path.join(args.fp_results_root, str(args.dataset_seed), 'id/test', f'{args.f0_predictions_base}_pp', 'pred'), args.quantile)
    print('tau:', tau)

    # save tau
    saved_taus[args.dataset_seed] = tau
    with open(args.tau_save_pth, 'wb') as f:
        pickle.dump(saved_taus, f)

    # print stats about tau
    n_above, n_total = calc_windows_above_threshold(os.path.join(args.fp_results_root, str(args.dataset_seed), 'id/test', f'{args.f0_predictions_base}_pp', 'pred'), tau)
    print(f'percentage of ID offline windows with ncs > tau: {round(n_above*100/n_total, 2)}')

    for mode in ["ood_0.0_3", "ood_0.0_5", "ood_0.9_0", "ood_1.0_0"]:
        print()
        
        n_above, n_total = calc_windows_above_threshold(os.path.join(args.fp_results_root, str(args.dataset_seed), mode, f'train_{tau}', f'{args.f0_predictions_base}_pp', 'pred'), tau)
        if n_total > 0:
            print(f'percentage of OOD {mode} train windows with ncs > tau: {round(n_above*100/n_total, 2)}, number of windows: {n_above}')

        n_above, n_total = calc_windows_above_threshold(os.path.join(args.fp_results_root, str(args.dataset_seed), mode, f'test_{tau}', f'{args.f0_predictions_base}_pp', 'pred'), tau)
        if n_total > 0:
            print(f'percentage of OOD {mode} test windows with ncs > tau: {round(n_above*100/n_total, 2)}, number of windows: {n_above}')