'''
Test is all on original model (non-finetuned)
'''
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Data split to use.')

    # predictions
    parser.add_argument('--gt_pred', default='pred', choices=['gt','pred'], help='Evaluate on groundtruth or prediction data.')
    parser.add_argument('--predictions_base', type=str, default='AgentFormer', help='Base path to model predictions.')
    
    # failure prediction parameters
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
    
    # incremental learning
    parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location of saved selected taus.')
    parser.add_argument('--use_incremental', action='store_true', help='Use arg to use incremental learning.')
    parser.add_argument('--cluster_method', default='kmeans', choices=['mems','kmeans'], help='Cluster method.')

    # uncertainty quantification
    parser.add_argument('--epsilon', type=float, default=0.03, help='RCP epsilon')    

    args = parser.parse_args()

    print(args)
    for scenario in ['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5']:
        if args.use_incremental and scenario == 'id':
            continue

        coverage = []
        for dataset_seed in range(10):

            # update predictions_base argument based on best epoch
            if not args.use_incremental:
                base_pth = os.path.join(args.predictions_base, f'results_s{dataset_seed}/racetrack_agentformer_pre')
            else:
                base_pth = os.path.join(args.predictions_base, f'results_s{dataset_seed}/racetrack_{scenario}_fttest')
            metrics_file = os.path.join(base_pth, 'validation_metrics.pkl')
            with open(metrics_file, 'rb') as f:
                metrics = pickle.load(f)
            best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
            predictions_base = os.path.join(base_pth, f'results/epoch_{best_epoch:04d}')

            # get results directory
            fp_results_dir = make_fp_results_dir(args.fp_results_root, dataset_seed, scenario, args.split, predictions_base, args.gt_pred,
                                                    args.use_incremental, args.cluster_method,
                                                    'rcp', None, args.epsilon)

            # load uncertainty quantification info
            if not os.path.exists(os.path.join(fp_results_dir, 'acp_data.pkl')):
                print('Experiment not done yet')
                break
            with open(os.path.join(fp_results_dir, 'acp_data.pkl'), 'rb') as f:
                acp_info = pickle.load(f)

            gamma = acp_info.pop('gamma')
            delta = acp_info.pop('delta')

            episodes = list(acp_info.keys()) # aka sequences

            ep_lens = []
            for e in episodes:
                far = acp_info[e].pop('far')
                mar = acp_info[e].pop('mar')
                fp = acp_info[e].pop('false_pos')
                tn = acp_info[e].pop('true_neg')
                fn = acp_info[e].pop('false_neg')
                tp = acp_info[e].pop('true_pos')
                steps_metric = acp_info[e].pop('steps')

                timesteps = list(acp_info[e].keys())
                ep_lens.append(len(timesteps))

                rhos = np.array([acp_info[e][t]['rho'] for t in timesteps])
                rho_hats = np.array([acp_info[e][t]['rho_hat'] for t in timesteps])
                C_ts = np.array([acp_info[e][t]['C_t'] for t in timesteps])
                
                ncs = rho_hats - rhos
                coverage += list((ncs <= C_ts))
        empirical_coverage = np.mean(coverage)
        print(f'{scenario}: {empirical_coverage:.2f}')