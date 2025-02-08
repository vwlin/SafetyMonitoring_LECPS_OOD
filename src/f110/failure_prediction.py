import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pickle

from utils.utils import compute_robustness, calc_C_t, make_fp_results_dir, calculate_delta_n, calculate_delta_tilde

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario',
                        default='id',
                        choices=['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='id or ood scenario. default id')
    
    # data
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Data split to use.')
    parser.add_argument('--dataset_base', type=str, default='data', help='Base path to dataset.')
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    
    # predictions
    parser.add_argument('--gt_pred', default='pred', choices=['gt','pred'], help='Evaluate on groundtruth or prediction data.')
    parser.add_argument('--predictions_base', type=str, default='models', help='Base path to model predictions.')
    
    # failure prediction parameters
    parser.add_argument('--safety_threshold', type=float, default=0.3, help='Min distance from walls.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
    
    # incremental learning
    parser.add_argument('--tau_save_pth', type=str, default=None, help='Location of saved selected taus. None to skip saving windows.')
    parser.add_argument('--use_incremental', action='store_true', help='Use arg to use incremental learning.')
    parser.add_argument('--cluster_method', default='mems', choices=['mems','kmeans'], help='Cluster method.')
    parser.add_argument('--cluster_metric', default='frechet', choices=['frechet','euclidean'], help='Distance metric for clustering.')
    parser.add_argument('--f0_predictions_base', type=str, default='models', help='Base path to non-finetuned model predictions.')

    # adaptive conformal prediction
    parser.add_argument('--pred_method', default='acp', choices=['point', 'cp', 'rcp', 'acp'], help='Use point prediction, conformal prediction, or adaptive conformal prediction.')
    parser.add_argument('--adapt_late', action='store_true', help='Use arg to adapt late.')
    parser.add_argument('--gamma', type=float, default=0.005, help='ACP gamma.')  # value from Adaptive Conformal Inference Under Distribution Shift paper
    parser.add_argument('--delta', type=float, default=0.1, help='ACP delta') # value from Adaptive Conformal Inference Under Distribution Shift paper
    parser.add_argument('--epsilon', type=float, default=0.08, help='RCP epsilon')    
    args = parser.parse_args()
    print(args)

    if args.use_incremental and args.scenario == 'id':
        raise NotImplementedError
        
    
    # load tau values
    if args.tau_save_pth is not None:
        assert os.path.exists(args.tau_save_pth), "Tau save path does not exist."
        with open(args.tau_save_pth, 'rb') as f:
            saved_taus = pickle.load(f)
        tau = saved_taus[args.dataset_seed]
    else:
        tau = None
    
    # make results directory
    fp_results_dir = make_fp_results_dir(args.fp_results_root, args.dataset_seed, args.scenario, args.split,
                                            args.predictions_base, args.gt_pred, args.use_incremental, args.cluster_method,
                                            args.pred_method, tau, args.epsilon)

    # load dataset with crash labels
    dataset_pth = os.path.join(args.dataset_base, str(args.dataset_seed), args.scenario, f'{args.split}.pkl')
    with open(dataset_pth, 'rb') as f:
        dataset = pickle.load(f)

    # load predictions
    predictions_pth = os.path.join(args.predictions_base, str(args.dataset_seed), 'eval', args.scenario, f'{args.split}_predictions.pkl')
    with open(predictions_pth, 'rb') as f:
        predictions = pickle.load(f)
    seq_eval = predictions.keys()

    # load calibration set for baseline uncertainty quantification methods
    if args.pred_method in ['cp', 'rcp']:
        with open(f'calibration_sets/calibration_set_{args.dataset_seed}.pkl', 'rb') as f:
            cal_set = pickle.load(f)

    # load memories and f1 predictions for incremental learning
    if args.use_incremental:
        if args.cluster_method == 'kmeans':
            with open(os.path.join(args.predictions_base, str(args.dataset_seed), 'pseudo_memories.pkl'), 'rb') as f:
                pseudo_memories = pickle.load(f)
            kmeans = pseudo_memories['kmeans']
            ood_clusters = pseudo_memories['ood_clusters']
        else: # mems
            raise NotImplementedError

        f0_predictions_pth = os.path.join(args.f0_predictions_base, str(args.dataset_seed), 'eval', args.scenario, f'{args.split}_predictions.pkl')
        with open(f0_predictions_pth, 'rb') as f:
            f0_predictions = pickle.load(f)
        assert seq_eval == f0_predictions.keys()

    # loop through each test episode (i.e., sequence)
    fail_pred, act_fa_rate, act_ma_rate, num_steps_early = [], [], [], []
    false_neg, true_neg, false_pos, true_pos = [], [], [], []
    precision, recall = [], []
    window_data_for_IL = {'X':[], 'Y':[], 'crash_windows':[]}
    acp_info = {'gamma':args.gamma, 'delta':args.delta}
    for seq_idx, seq_name in enumerate(seq_eval):
        print(f'\n\n----------evaluating sequence {seq_name}----------')

        current_timestep = args.n_history-1 # initializing current_timestep with zero indexing
        delta_t = args.delta

        gt_robustness = []
        pred_robustness = []
        ncs = [] # pred_robustness - gt_robustness
        alarm = np.array([])
        err = []

        # load gt crash label data
        assert seq_name in dataset
        crashes = list(dataset[seq_name]['crash_labels'])
        if True in crashes:
            end_idx = crashes.index(True) + args.n_horizon
            crashes = crashes[:end_idx]
            print(f'last crash index: {end_idx-1}')
        print(crashes)

        if len(crashes) - args.n_history - args.n_horizon < (args.n_history+2*args.n_horizon): # low_step_cutoff
            print(f'insufficient length episode. skipping seq {seq_name:s}')
            num_steps_early.append(np.NaN)
            continue

        # load prediction data for this sequence
        seq_predictions = predictions[seq_name]
        history, horizon, horizon_pred =  seq_predictions['X'], seq_predictions['Y'], seq_predictions['Y_pred']
        assert history.shape[0] == horizon.shape[0] == horizon_pred.shape[0]
        n_windows = history.shape[0]
        if args.use_incremental:
            assert n_windows == f0_predictions[seq_name]['X'].shape[0] == f0_predictions[seq_name]['Y'].shape[0] == f0_predictions[seq_name]['Y_pred'].shape[0]

        # loop through each window
        windows_for_incremental_learning = {} # record every window's ncs and history
        running_sum = []
        acp_info[seq_name] = {}
        for w in range(n_windows):
            print(f'\nevaluating seq {seq_name:s}, forecasting frame {current_timestep+1:06d} to {current_timestep+args.n_horizon:06d}')
            hist, horz, horz_pred = history[w,:,0:2], horizon[w,:,:], horizon_pred[w,:,:]
            assert(horz.shape == horz_pred.shape)
            print('history:')
            print(hist)

            # start algorithm at a delay to ensure there are enough NCS to calculate a C_t value
            predictions_start = (args.n_history+2*args.n_horizon-1)

            if args.adapt_late: # when to start adapting conformal prediction by updating alpha_t
                adapt_start = predictions_start
            else:
                adapt_start = (args.n_history+args.n_horizon-1) # 0
            last_real_timestep = current_timestep-(args.n_history+args.n_horizon-1) # our algorithm processes in "the future" in simulation. But in a real system, we can only calculated NCS once the prediction period has passed. (prev_timesteps_req-1 accounts for offset of starting time at prev_timesteps_req, future_timesteps accounts for how long you need to wait to have seen the GT for the entire predicted trajectory)

            # calculate robustness
            if args.use_incremental:
                if args.cluster_method == 'kmeans':
                    # use pseudo memories (i.e., clusters)
                    assert np.all(history[w,:,:] == f0_predictions[seq_name]['X'][w,:,:])
                    cluster = kmeans.predict(history[w,:,:].reshape(-1,args.n_history*3).astype('float'))
                    if cluster not in ood_clusters: # replace horz pred with pred from f0
                        print('f0 predictor selected')
                        horz_pred = f0_predictions[seq_name]['Y_pred'][w,:,:]
                    else:
                        print('f1 predictor selected')
                else:
                    raise NotImplementedError

            print('horizon groundtruth:')
            print(horz)
            print('horizon prediction:')
            print(horz_pred)

            if args.gt_pred == 'pred':
                pred_robustness_val = compute_robustness(trajectories=horz_pred, safety_threshold=args.safety_threshold)
            else:
                pred_robustness_val = compute_robustness(trajectories=horz, safety_threshold=args.safety_threshold)
            gt_robustness_val = compute_robustness(trajectories=horz, safety_threshold=args.safety_threshold)
            ###
            
            pred_robustness.append(pred_robustness_val)
            gt_robustness.append(gt_robustness_val)

            # update ncs array
            ncs.append(pred_robustness_val - gt_robustness_val)

            print(f'ground truth robustness value: {gt_robustness[-1]}')
            print(f'predicted robustness value: {pred_robustness[-1]}')
            print(f'NCS: {ncs[-1]}')

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
                    running_sum.append(err_t)
                    print(f'C_t: {C_t}')
                    print(f'err_t: {err_t}')
                    print(f'delta_t: {delta_t}')
                    print(f'average miscoverage rate: {np.mean(running_sum)}')
                else:
                    print('abstain')

                if current_timestep > predictions_start:
                    print(f'crash in next {args.n_horizon} steps? {relevant_crash_labels[-args.n_horizon:]}')
                    print(f'alarm? {alarm[-1]}')

                # save data for use with incremental learning part of pipeline
                windows_for_incremental_learning[current_timestep] = {
                    'ncs': ncs[-1],
                    'C_t': C_t,
                }
                if args.tau_save_pth is not None and np.abs(ncs[-1]) > tau:
                    window_data_for_IL['X'].append(history[w,:,:])
                    window_data_for_IL['Y'].append(horizon[w,:,:])
                    window_data_for_IL['crash_windows'].append(np.any(relevant_crash_labels[-args.n_horizon:]))
            
            if(np.sum(list(crashes[current_timestep+1 : current_timestep+1+args.n_horizon])) == args.n_horizon):
                break

            current_timestep += 1
        
        # dump data for incremental learning
        with open(os.path.join(fp_results_dir, f'{seq_name}_ood_windows.pkl'), 'wb') as f:
            pickle.dump(windows_for_incremental_learning, f)

        print("Err:", err)
        print("Alarm: ", alarm)
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

    # save information to analyze convergence rate of ACP, coverage of UQ methods, etc
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

    # save high-error windows for incremental learning
    if args.tau_save_pth is not None:
        window_data_for_IL['X'] = np.stack(window_data_for_IL['X'])
        window_data_for_IL['Y'] = np.stack(window_data_for_IL['Y'])
        window_data_for_IL['crash_windows'] = np.array(window_data_for_IL['crash_windows'], dtype=bool)

        with open(os.path.join(fp_results_dir, 'window_data.pkl'), 'wb') as f:
            pickle.dump(window_data_for_IL, f)