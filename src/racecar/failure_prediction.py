import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from AgentFormer.utils.utils import find_unique_common_from_lists, load_list_from_folder, load_txt_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario',
                        default='id',
                        choices=['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5'],
                        help='id or ood scenario. default id')

    # data
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Data split to use.')
    parser.add_argument('--dataset_base', type=str, default='AgentFormer/datasets/racetrack', help='Base path to dataset.')
    parser.add_argument('--dataset_seed', type=int, default=0, help='Random seed for dataset (trial number).')

    # predictions
    parser.add_argument('--gt_pred', default='pred', choices=['gt','pred'], help='Evaluate on groundtruth or prediction (reconstructed) data.')
    parser.add_argument('--predictions_base', type=str, default='AgentFormer/results_s0/racetrack_agentformer_pre', help='Base path to model predictions.')

    # failure prediction parameters
    parser.add_argument('--safety_threshold', type=float, default=5.4, help='Min distance from other vehicles.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')

    # incremental learning
    parser.add_argument('--tau_save_pth', type=str, default=None, help='Location of saved selected taus. None to skip saving windows.')
    parser.add_argument('--use_incremental', action='store_true', help='Use arg to use incremental learning.')
    parser.add_argument('--cluster_pth', type=str, help='Location of saved cluster data.')
    parser.add_argument('--cluster_method', default='kmeans', choices=['mems','kmeans'], help='Cluster method.')
    parser.add_argument('--cluster_metric', default='euclidean', choices=['frechet','euclidean'], help='Distance metric for clustering.')
    parser.add_argument('--f0_predictions_base', type=str, default='AgentFormer/results_s0/racetrack_vehicles_2_ogtest', help='Base path to non-finetuned model predictions.')

    # adaptive conformal prediction
    parser.add_argument('--pred_method', default='acp', choices=['point', 'cp', 'rcp', 'acp'], help='Use point prediction, conformal prediction, or adaptive conformal prediction.')
    parser.add_argument('--adapt_late', action='store_true', help='Use arg to adapt late.')
    parser.add_argument('--gamma', type=float, default=0.005, help='ACP gamma.')  # value from Adaptive Conformal Inference Under Distribution Shift paper
    parser.add_argument('--delta', type=float, default=0.1, help='ACP delta') # value from Adaptive Conformal Inference Under Distribution Shift paper
    parser.add_argument('--epsilon', type=float, default=0.03, help='RCP epsilon')

    # miscellaneous
    parser.add_argument('--verbose', action='store_true', help='Use arg to print extra details.')

    args = parser.parse_args()
    print(args)

    if args.use_incremental and args.scenario == 'id':
        raise NotImplementedError
    if args.cluster_method == 'mems' or args.cluster_metric == 'frechet':
        raise NotImplementedError

    if args.scenario == 'id':
        n_agents = 2
    else:
        n_agents = int(args.scenario.split('_')[1]) + 1

    # load tau values
    if args.tau_save_pth is not None:
        assert os.path.exists(args.tau_save_pth), "Tau save path does not exist."
        with open(args.tau_save_pth, 'rb') as f:
            saved_taus = pickle.load(f)
        tau = saved_taus[args.dataset_seed]
    else:
        tau=None

    # update predicitons_base argument based on best epoch
    metrics_file = os.path.join(args.predictions_base, 'validation_metrics.pkl')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
    args.predictions_base = os.path.join(args.predictions_base, f'results/epoch_{best_epoch:04d}')

    f0_metrics_file = os.path.join(args.f0_predictions_base, 'validation_metrics.pkl')
    with open(f0_metrics_file, 'rb') as f:
        f0_metrics = pickle.load(f)
    f0_best_epoch = min(f0_metrics, key=lambda x: f0_metrics[x]['ADE'])
    args.f0_predictions_base = os.path.join(args.f0_predictions_base, f'results/epoch_{f0_best_epoch:04d}')

    # make results directory
    fp_results_dir = make_fp_results_dir(args.fp_results_root, args.dataset_seed, args.scenario, args.split, args.predictions_base, args.gt_pred,
                                            args.use_incremental, args.cluster_method,
                                            args.pred_method, tau, args.epsilon)

    # prepare to load groundtruth
    dataset_pth = os.path.join(args.dataset_base, f'{args.scenario}_s{str(args.dataset_seed)}')
    seq_train, seq_val, seq_test = get_racetrack_split_for_fp(dataset_pth)
    seq_eval = globals()[f'seq_{args.split}']

    # get dict of ego vehicle IDs
    with open(os.path.join(dataset_pth, f'ego_ids_{args.split}.pkl'), 'rb') as f:
        ego_ids_dict = pickle.load(f)

    # prepare dataset directory to save incremental learning data to
    if tau is not None:
        window_dir = os.path.join(args.dataset_base, f'{args.scenario}_windows_s{args.dataset_seed}')
        if os.path.exists(window_dir):
            print(f'Warning! {window_dir} directory already exists. New data will be added to directory, and old data will be preserved.')
        os.makedirs(window_dir, exist_ok=True)
        tr_val_split = [0.75, 0.25]

    # load calibration set for baseline uncertainty quantification methods
    if args.pred_method in ['cp', 'rcp']:
        with open(f'calibration_sets/calibration_set_{args.dataset_seed}.pkl', 'rb') as f:
            cal_set = pickle.load(f)

    # load memories for incremental learning
    if args.use_incremental:
        with open(args.cluster_pth, 'rb') as f:
            pseudo_memories = pickle.load(f)
        kmeans = pseudo_memories['kmeans']
        ood_clusters = pseudo_memories['ood_clusters']

    # loop through each test episode (i.e., sequence)
    fail_pred, act_fa_rate, act_ma_rate, num_steps_early = [], [], [], []
    false_neg, true_neg, false_pos, true_pos = [], [], [], []
    precision, recall = [], []
    acp_info = {'gamma':args.gamma, 'delta':args.delta}
    for seq_idx, seq_name in enumerate(seq_eval):
        print(f'\n\n----------evaluating sequence {seq_name}----------')
        
        current_timestep = args.n_history-1 # initializing current_timestep with zero indexing
        delta_t = args.delta

        gt_robustness, pred_robustness, ncs, err = [], [], [], []
        alarm = np.array([])

        # load gt raw data and crash labels from datasets folder
        gt_raw, gt_raw_all, crashes = load_groundtruth_data(dataset_pth, seq_name)

        # truncate crash labels to contain only up to first crash plus n_horizon points
        if True in crashes:
            end_idx = crashes.index(True) + args.n_horizon
            crashes = crashes[:end_idx]
            if args.verbose: print(f'last crash index: {end_idx-1}')

        # check that length of episode is not too short for algorithm
        if len(crashes) - args.n_history - args.n_horizon < (args.n_history+2*args.n_horizon):
            print(f'insufficient length episode. skipping seq {seq_name:s}')
            num_steps_early.append(np.NaN)
            continue

        # pull the text files containing AgentFormer window predictions
        if args.gt_pred == 'pred': gt_recon = 'recon'
        else: gt_recon = 'gt'
        print(os.path.join(args.predictions_base, args.split, gt_recon, seq_name))
        data_filelist, _ = load_list_from_folder(os.path.join(args.predictions_base, args.split, gt_recon, seq_name))

        if args.use_incremental:
            print(os.path.join(args.f0_predictions_base, args.split, gt_recon, seq_name))
            f0_data_filelist, _ = load_list_from_folder(os.path.join(args.f0_predictions_base, args.split, gt_recon, seq_name))
            assert len(data_filelist) == len(f0_data_filelist)

        # loop through each window e.g., seq_0001 - frame_000009
        ncs_for_incremental_learning = {} # record every window's ncs and history
        acp_info[seq_name] = {}
        for (d,data_file) in enumerate(data_filelist):
            print(f'\nevaluating seq {seq_name:s}, forecasting frame {current_timestep+1:06d} to {current_timestep+args.n_horizon:06d}')

            # start algorithm at a delay to ensure there are enough NCS to calculate a C_t value
            predictions_start = (args.n_history+2*args.n_horizon-1)

            # determine when to start adapting conformal prediction by updating alpha_t
            if args.adapt_late:
                adapt_start = predictions_start
            else:
                adapt_start = (args.n_history+args.n_horizon-1) # 0

            # our algorithm processes in "the future" in simulation, but in a real system we can only calculated NCS once the prediction period has passed. (prev_timesteps_req-1 accounts for offset of starting time at prev_timesteps_req, future_timesteps accounts for how long you need to wait to have seen the GT for the entire predicted trajectory)
            last_real_timestep = current_timestep-(args.n_history+args.n_horizon-1)

            # load reconstructed data from AgentFormer, and process reconstructed and groundtruth data
            ego_id = float(ego_ids_dict[int(seq_name.split('_')[1])])
            horz_gt, horz_pred = process_reconstructed_and_gt_data(data_file, gt_raw, ego_id)
            
            if args.use_incremental:
                # if using incremental learning, load reconstructed data from finetuned AgentFormer
                f0_horz_gt, f0_horz_pred = process_reconstructed_and_gt_data(f0_data_filelist[d], gt_raw, ego_id)
                assert (f0_horz_gt == horz_gt).all()
                
                # grab history for clustering
                hist_raw = gt_raw_all[gt_raw_all[:, 0].astype('int') >= current_timestep-args.n_history+1]
                hist_raw = hist_raw[hist_raw[:, 0].astype('int') <= current_timestep]
                
                hist_ids = np.unique(hist_raw[:, 1])
                hist_steps = np.unique(hist_raw[:, 0])
                
                hist = []
                for s in hist_steps:
                    s_data = hist_raw[hist_raw[:,0] == s][:,[13, 15, 16]].astype('float64')
                    s_data = np.pad(s_data, ((0,n_agents-len(hist_ids)),(0,0)), 'constant', constant_values=(200, 200))
                    hist.append(s_data)
                hist = np.swapaxes(np.stack(hist), 0, 1).astype('float64')
                assert hist.shape[1] == args.n_history

                # sort history agents based on initial distance from origin
                predicate = sort_metric(hist[:,0,0:2])
                order = np.argsort(predicate)
                hist = hist[order,:,:]

                # cluster based on history
                cluster = kmeans.predict(hist.reshape(-1,n_agents*args.n_history*3).astype('float'))
                if cluster not in ood_clusters: # replace horz pred with pred from f0
                    print('f0 predictor selected')
                    horz_pred = f0_horz_pred
                else:
                    print('f1 predictor selected')

            if args.verbose:
                print('horizon groundtruth:')
                print(horz_gt)
                print('horizon prediction:')
                print(horz_pred)

            # calculate robustness
            if args.gt_pred == 'pred':
                pred_robustness_val = compute_robustness(trajectories=horz_pred, safety_threshold=args.safety_threshold)
            else:
                pred_robustness_val = compute_robustness(trajectories=horz_gt, safety_threshold=args.safety_threshold)
            gt_robustness_val = compute_robustness(trajectories=horz_gt, safety_threshold=args.safety_threshold)
            
            pred_robustness.append(pred_robustness_val)
            gt_robustness.append(gt_robustness_val)

            # update ncs array
            ncs.append(pred_robustness_val - gt_robustness_val)

            if args.verbose:
                print(f'ground truth robustness value: {gt_robustness[-1]}')
                print(f'predicted robustness value: {pred_robustness[-1]}')
                print(f'NCS: {ncs[-1]}')

            # predict failures if enough startup time has gone by
            if current_timestep > predictions_start:
                if current_timestep == predictions_start + 1:
                    relevant_crash_labels = list(crashes[current_timestep+1 : current_timestep+1+args.n_horizon])
                elif current_timestep > predictions_start + 1:
                    relevant_crash_labels.append(crashes[current_timestep+args.n_horizon])

                # determine whether to alarm
                if last_real_timestep >= 0:
                    if args.pred_method == 'cp':
                        visible_ncs = ncs[0:last_real_timestep+1]
                        C_t = calc_C_t(existing_scores=cal_set, t=len(cal_set)-1, delta_t=args.delta) # t starting from 0
                    elif args.pred_method == 'rcp':
                        visible_ncs = ncs[0:last_real_timestep+1]
                        delta_n = calculate_delta_n(args.delta, len(cal_set), args.epsilon)
                        delta_tilde = calculate_delta_tilde(delta_n, args.epsilon)
                        p = int(np.ceil((len(cal_set)) * (1 - delta_tilde)))
                        C_t = cal_set[p - 1]
                    else:
                        visible_ncs = ncs[0:last_real_timestep+1] # ncs can only be used for seen timesteps for gt data
                        C_t = calc_C_t(existing_scores=visible_ncs, t=current_timestep-adapt_start-1, delta_t=delta_t) # t starting from 0
                else:
                    C_t = None

                acp_info[seq_name][current_timestep] = {
                    'rho': gt_robustness_val,
                    'rho_hat': pred_robustness_val,
                    'C_t': C_t
                }

                if C_t is None or pred_robustness_val is None: # not enough data to make predictions
                    alarm = np.append(alarm, -1)
                elif args.pred_method == 'point': # point prediction
                    if pred_robustness_val < 0:
                        alarm = np.append(alarm, 1)
                    else:
                        alarm = np.append(alarm, 0)
                else: # conformal prediction or adaptive conformal prediction
                    if pred_robustness_val < C_t:
                        alarm = np.append(alarm, 1)
                    else:
                        alarm = np.append(alarm, 0)

                # update conformal prediction
                if C_t is not None: # not enough data to make predictions
                    err_t = int(visible_ncs[-1] > C_t)
                    err.append(err_t)
                    if args.pred_method == 'acp':
                        delta_t = delta_t+args.gamma*(args.delta-err_t)
                    # running_sum.append(err_t)
                    if args.verbose:
                        print(f'C_t: {C_t}')
                        print(f'err_t: {err_t}')
                        print(f'delta_t: {delta_t}')
                        # print(f'average miscoverage rate: {np.mean(running_sum)}')
                else:
                    print('abstain')

                print(f'crash in next {args.n_horizon} steps? {relevant_crash_labels[-args.n_horizon:]}')
                print(f'alarm? {alarm[-1]}')

                # save data for use with incremental learning part of pipeline
                ncs_for_incremental_learning[current_timestep] = {
                    'ncs': ncs[-1],
                    'C_t': C_t,
                }
                if tau is not None and np.abs(ncs[-1]) > tau:
                    gt_this_window = gt_raw_all[gt_raw_all[:, 0].astype('int') >= current_timestep-args.n_history+1]
                    gt_this_window = gt_this_window[gt_this_window[:, 0].astype('int') <= current_timestep+args.n_horizon]

                    window_str = []
                    for i in range(len(gt_this_window)):
                        window_str.append(' '.join(gt_this_window[i,:]))
                    window_str = '\n'.join(window_str)

                    if np.random.uniform() < tr_val_split[0]:
                        window_split = 'train'
                    else:
                        window_split = 'val'

                    new_seq_name = seq_name.replace('_train', '')
                    new_seq_name = new_seq_name.replace('_val', '')

                    window_filename = f'{new_seq_name}_{os.path.splitext(os.path.basename(data_file))[0]}_{window_split}.txt'

                    with open(os.path.join(window_dir, window_filename), 'w') as f:
                        f.write(window_str)

            if(np.sum(list(crashes[current_timestep+1 : current_timestep+1+args.n_horizon])) == args.n_horizon):
                break

            current_timestep += 1

        # dump data for incremental learning and for tau selection
        with open(os.path.join(fp_results_dir, f'{seq_name}_ood_windows.pkl'), 'wb') as f:
            pickle.dump(ncs_for_incremental_learning, f)

        if args.verbose:
            print("Err:", err)
            print("Alarm: ", list(alarm))

        if len(alarm)>0:
            fail_pred.append(int(max(alarm)))

            # collect stats
            n_no_crashes = len([i for i in range(len(alarm)) if not np.any(relevant_crash_labels[i:i+args.n_horizon])])
            n_crashes = len([i for i in range(len(alarm)) if np.any(relevant_crash_labels[i:i+args.n_horizon])])

            if n_no_crashes > 0:
                # calculate false positives (false alarms)
                false_alarms_idx = [i for i in range(len(alarm)) if alarm[i]==1 and not np.any(relevant_crash_labels[i:i+args.n_horizon])]
                act_fa_rate.append(len(false_alarms_idx)/n_no_crashes)
                print('False alarm rate:', act_fa_rate[-1])
                false_pos.append(len(false_alarms_idx))

                # calculate true negatives
                true_neg.append(n_no_crashes - len(false_alarms_idx)) 
            else:
                print('False alarm rate: N/A')
                act_fa_rate.append(np.NaN)
                false_pos.append(np.NaN)
                true_neg.append(np.NaN)

            if n_crashes > 0:
                # calculate false negatives (missed alarm)
                missed_alarms_idx = [i for i in range(len(alarm)) if alarm[i]==0 and np.any(relevant_crash_labels[i:i+args.n_horizon])]
                act_ma_rate.append(len(missed_alarms_idx)/n_crashes)
                print('Missed alarm rate:', act_ma_rate[-1])
                false_neg.append(len(missed_alarms_idx))

                # calculate true positives
                true_pos.append(n_crashes - len(missed_alarms_idx))
            else:
                print('Missed alarm rate: N/A')
                act_ma_rate.append(np.NaN)
                false_neg.append(np.NaN)
                true_pos.append(np.NaN)

            if n_crashes > 0:
                early_steps = [relevant_crash_labels[i:i+args.n_horizon].index(1) + 1 + i - i\
                                for i in range(len(alarm)) if alarm[i]==1 and np.any(relevant_crash_labels[i:i+args.n_horizon])]
                # there can only be one crash. all crash labels after the crash are 1 consecutively, so there could be more than 1 crash label
                # we only care about how early we correctly alarm before the first crash label
                if len(early_steps) > 0:
                    num_steps_early.append(early_steps[0])
                    print('Number of steps prior to crash we raise alarm:', num_steps_early[-1])
                else:
                    num_steps_early.append(np.NaN)
                    print('Number of steps prior to crash we raise alarm: N/A')
            else:
                num_steps_early.append(np.NaN)
                print('Number of steps prior to crash we raise alarm: N/A')

        else:
            act_fa_rate.append(np.NaN)
            act_ma_rate.append(np.NaN)

            false_pos.append(np.NaN)
            true_neg.append(np.NaN)
            false_neg.append(np.NaN)
            true_pos.append(np.NaN)

            num_steps_early.append(np.NaN)

        if ~np.isnan(true_pos[-1]) and ~np.isnan(false_pos[-1]) and (true_pos[-1] + false_pos[-1]) > 0:
            precision.append(true_pos[-1] / (true_pos[-1] + false_pos[-1]))
        else:
            precision.append(np.NaN)
        if ~np.isnan(true_pos[-1]) and ~np.isnan(false_neg[-1]) and (true_pos[-1] + false_neg[-1]) > 0:
            recall.append(true_pos[-1] / (true_pos[-1] + false_neg[-1]))
        else:
            recall.append(np.NaN)

        acp_info[seq_name]['far'] = act_fa_rate[-1]
        acp_info[seq_name]['mar'] = act_ma_rate[-1]

        acp_info[seq_name]['false_pos'] = false_pos[-1]
        acp_info[seq_name]['true_neg'] = true_neg[-1]
        acp_info[seq_name]['false_neg'] = false_neg[-1]
        acp_info[seq_name]['true_pos'] = true_pos[-1]

        acp_info[seq_name]['steps'] = num_steps_early[-1]

    # save information to analyze convergence rate of ACP
    with open(os.path.join(fp_results_dir, 'acp_data.pkl'), 'wb') as f:
        pickle.dump(acp_info, f)

    # save per-episode results
    episode_results = {
        'FAR':act_fa_rate,
        'MAR':act_ma_rate,
        'false_pos':false_pos,
        'true_neg':true_neg,
        'false_neg':false_neg,
        'true_pos':true_pos,
        'steps':num_steps_early,
        'precision':precision,
        'recall':recall
    }
    with open(os.path.join(fp_results_dir, 'episode_results.pkl'), 'wb') as f:
        pickle.dump(episode_results, f)
    print("\nFailure Prediction Result on all test sequences:", fail_pred)

    # save overall results, averaged over episodes
    print("\nAverages across all sequences:")
    results = {}
    if np.count_nonzero(~np.isnan(act_fa_rate)) > 0:
        print('False alarm rate:', np.nanmean(act_fa_rate))
        results['FAR'] = np.nanmean(act_fa_rate)
    else:
        print('False alarm rate: N/A')
        results['FAR'] = np.NAN

    if np.count_nonzero(~np.isnan(act_ma_rate)) > 0:
        print('Missed alarm rate:', np.nanmean(act_ma_rate))
        results['MAR'] = np.nanmean(act_ma_rate)
    else:
        print('Missed alarm rate: N/A')
        results['MAR'] = np.NAN

    # precision - what proportion of positive ids was actually correct
    if np.count_nonzero(~np.isnan(precision)) > 0:
        print('Precision:', np.nanmean(precision))
        results['precision'] = np.nanmean(precision)
    else:
        print('Precision: N/A')
        results['precision'] = np.NAN
    
    # recall - what proportion of actual positives was identified correctly
    if np.count_nonzero(~np.isnan(recall)) > 0:
        print('Recall:', np.nanmean(recall))
        results['recall'] = np.nanmean(recall)
    else:
        print('Recall: N/A')
        results['recall'] = np.NAN

    print('Number of steps prior to crash we raise alarm:')
    for (s,seq_name) in enumerate(seq_eval):
        print(seq_name, ':', num_steps_early[s])
    if np.count_nonzero(~np.isnan(num_steps_early)) > 0:
        print('Average:', np.nanmean(num_steps_early))
        results['steps'] = np.nanmean(num_steps_early)
    else:
        print('Average: N/A')
        results['steps'] = np.NAN

    with open(os.path.join(fp_results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)