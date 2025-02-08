import numpy as np
import pickle
import os
import argparse

from utils.utils import make_fp_results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario',
                        default='id',
                        choices=['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='id or ood scenario. default id')
    parser.add_argument('--summary_results_pth', type=str, default='results_summary.pkl', help='Location of results summary.')
    
    # data
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Data split to use.')
    
    # predictions
    parser.add_argument('--gt_pred', default='pred', choices=['gt','pred'], help='Evaluate on groundtruth or prediction data.')
    parser.add_argument('--predictions_base', type=str, default='models', help='Base path to model predictions.')
    
    # failure prediction parameters
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
    
    # incremental learning
    parser.add_argument('--tau', action='store_true', help='Use arg to pull data from tau runs.')
    parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location of saved selected taus.')
    parser.add_argument('--use_incremental', action='store_true', help='Use arg to use incremental learning.')
    parser.add_argument('--cluster_method', default='mems', choices=['mems','kmeans'], help='Cluster method.')
    parser.add_argument('--cluster_metric', default='frechet', choices=['frechet','euclidean'], help='Distance metric for clustering.')

    # uncertainty quantification
    parser.add_argument('--pred_method', default='acp', choices=['point', 'cp', 'rcp', 'acp'], help='Use point prediction, conformal prediction, or adaptive conformal prediction.')
    parser.add_argument('--epsilon', type=float, default=0.09, help='RCP epsilon')    
    
    args = parser.parse_args()

    print(args)

    # load or create new results summary dictionary
    if os.path.exists(args.summary_results_pth):
        with open(args.summary_results_pth, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    # load tau values
    if args.tau:
        assert os.path.exists(args.tau_save_pth), "Tau save path does not exist."
        with open(args.tau_save_pth, 'rb') as f:
            saved_taus = pickle.load(f)

    # go through results
    far, mar, precision, recall, steps = {}, {}, {}, {}, {}
    for dataset_seed in range(1,11):
        # results dir path
        if args.tau:
            tau = saved_taus[dataset_seed]
        else:
            tau = None
        fp_results_dir = make_fp_results_dir(args.fp_results_root, dataset_seed, args.scenario, args.split,
                                            args.predictions_base, args.gt_pred, args.use_incremental, args.cluster_method,
                                            args.pred_method, tau, args.epsilon)

        with open(os.path.join(fp_results_dir, 'episode_results.pkl'), 'rb') as f:
            episode_results = pickle.load(f)
            
        far[dataset_seed] = episode_results['FAR']
        mar[dataset_seed] = episode_results['MAR']
        precision[dataset_seed] = episode_results['precision']
        recall[dataset_seed] = episode_results['recall']
        steps[dataset_seed] = episode_results['steps']

    # update results summary dictionary and dump
    if args.pred_method == 'rcp':
        key = f'{args.scenario}_{args.use_incremental}_{args.pred_method}_{args.epsilon}'
    else:
        key = f'{args.scenario}_{args.use_incremental}_{args.pred_method}'
    all_results[key] = {
        'far_list':far,
        'mar_list':mar,
        'precision_list':precision,
        'recall_list':recall,
        'steps_list':steps
    }
    with open(args.summary_results_pth, 'wb') as f:
        pickle.dump(all_results, f)