import numpy as np
from torch.utils.data import Dataset
import math
import torch
import os
import sys
import pickle

sys.path.append("..")
from AgentFormer.utils.utils import load_txt_file, isfile, find_unique_common_from_lists

'''
assuming safety property to be no collision with other vehicles, safety_threshold is the min
distance that we want the ego vehicle to maintain from all the other agents,
computes robustness at each future timestamp and returns min value of
robustess among these future timestamps
Parameters
    trajectories - ndarray(n_agents, n_horizon, 2)
'''
def compute_robustness(trajectories, safety_threshold):
    n_agents, n_steps, _ = trajectories.shape

    min_dist = np.Inf
    for t in range(n_steps): # iterating over future timesteps

        ego_pos = trajectories[0, t, :] # agent 0 is the ego vehicle

        for a in range(1, n_agents): # iterating over each other agent
            agent_j_pos = trajectories[a, t, :]

            diff = ego_pos - agent_j_pos
            dist = np.linalg.norm(diff)

            if dist < min_dist:
                min_dist = dist

    robustness_value = min_dist-safety_threshold
    return robustness_value

'''
existing_scores is a sorted array, latest score to be inserted at
the appropriate position in existing_scores array
'''
def calc_C_t(existing_scores, t, delta_t):
    sorted_scores = np.sort(existing_scores)
    C_t_index = int(math.ceil((t+1)*(1-delta_t)))

    if len(sorted_scores) > C_t_index-1: # sanity check on the len of visible ncs as we might not have enough due to abstain
        return sorted_scores[C_t_index-1]
    elif len(sorted_scores) > 0: # chking if there is at least one item in ncs arr
        return sorted_scores[-1]
    else:
        return None

'''
Make failure prediction results directory based on experiment parameters.
'''
def make_fp_results_dir(fp_results_root, dataset_seed, scenario, split, predictions_base, gt_pred,
                        use_incremental, cluster_method,
                        pred_method,
                        tau=None, epsilon=0.03):
    fp_results_dir = os.path.join(fp_results_root, str(dataset_seed), scenario, split)
    if tau != None:
        fp_results_dir = f'{fp_results_dir}_{str(tau)}'

    fp_results_dir = os.path.join(fp_results_dir, os.path.basename(predictions_base))
    if use_incremental:
        fp_results_dir = f'{fp_results_dir}_{cluster_method}'
        if cluster_method == 'mems':
            fp_results_dir = f'{fp_results_dir}-{cluster_method}'
        fp_results_dir = f'{fp_results_dir}-il'

    if pred_method == 'point': pred_method = 'pp'
    fp_results_dir = f'{fp_results_dir}_{pred_method}'
    if pred_method == 'rcp':
        fp_results_dir = f'{fp_results_dir}_{epsilon}'
    
    fp_results_dir = os.path.join(fp_results_dir, gt_pred)
    os.makedirs(fp_results_dir, exist_ok=True)

    return fp_results_dir

'''
Returns lists of all the episode data files in dataset_dir.
'''
def get_racetrack_split_for_fp(dataset_dir):
    files = os.listdir(dataset_dir)

    train, val, test = [], [], []
    for file in files:
        basename = os.path.splitext(file)[0]
        if 'labels' in basename or 'mapping' in basename or 'ego_ids' in basename:
            continue
        if 'train' in basename:
            train.append(basename)
        elif 'val' in basename:
            val.append(basename)
        else:
            test.append(basename)

    return train, val, test

'''
Load groundtruth seq_name data and crash labels from dataset_dir directory
'''
def load_groundtruth_data(dataset_dir, seq_name):
    # load each line of textfile as a string in a list
    gt_data, _ = load_txt_file(os.path.join(dataset_dir, seq_name+'.txt'))

    # convert to numpy array
    gt_raw, gt_raw_all = [], []
    for raw_line_data in gt_data:
        # for gt_raw, grab only the fields of interest (timestep, id, x, y) from each line
        line_data = np.array([raw_line_data.split(' ')])[:, [0, 1, 13, 15]][0].astype('float64')
        if line_data[1] == -1: continue
        gt_raw.append(line_data)

        # for gt_raw_all, grab all fields from each line
        line_data_all = np.array([raw_line_data.split(' ')])[0]
        gt_raw_all.append(line_data_all)
    gt_raw, gt_raw_all = np.stack(gt_raw), np.stack(gt_raw_all)

    # load crash label pickle file and convert data to list
    crash_label_file = seq_name.split('_')
    crash_label_file.insert(2, 'labels')
    crash_label_file = '_'.join(crash_label_file)
    crash_label_filepth = os.path.join(dataset_dir, f'{crash_label_file}.pkl')
    if os.path.exists(crash_label_filepth):
        with open(crash_label_filepth, 'rb') as f:
            crash_data = pickle.load(f)
        crashes = list(crash_data['crash_flags'])
    else:
        crashes = None

    return gt_raw, gt_raw_all, crashes

'''
Filter pred and gt data.
'''
def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new

'''
Load reconstructed window data from data_file.
Returns
    gt_traj - ndarray(n_agents, n_horizon, 2)
    pred_traj - ndarray(n_agents, n_horizon, 2)
'''
def process_reconstructed_and_gt_data(data_file, gt_raw, ego_id):
    # load as numpy if data_file exists
    if isfile(data_file): # for reconstruction or deterministic
        all_traj = np.loadtxt(data_file, delimiter=' ', dtype='float64')        # (frames x agents) x 4
        all_traj = np.expand_dims(all_traj, axis=0)                             # 1 x (frames x agents) x 4          
    elif isfolder(data_file): # for stochastic with multiple samples
        sample_list, _ = load_list_from_folder(data_file)
        sample_all = []
        for sample in sample_list:
            sample = np.loadtxt(sample, delimiter=' ', dtype='float64')        # (frames x agents) x 4
            sample_all.append(sample)
        all_traj = np.stack(sample_all, axis=0)                                # samples x (framex x agents) x 4
    else:
        assert False, 'error'

    # grab list of ids and frames
    id_list = np.unique(all_traj[:, :, 1])
    frame_list = np.unique(all_traj[:, :, 0])

    # ensure ego ID is the first in the ID list
    ego_idx = np.where(id_list==ego_id)[0][0]
    id_list = np.delete(id_list, ego_idx)
    id_list = np.insert(id_list, 0, ego_id)

    # loop through all objects (each id)
    agent_traj = []
    gt_traj = [] # list of numpies
    for idx in id_list:
        # grab all groundtruth frames with same id
        gt_idx = gt_raw[gt_raw[:, 1] == idx]                          # frames x 4

        # grab all predicted frames with same id
        ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
        pred_idx = all_traj[:, ind, :]                                # sample x frames x 4
        
        # filter data
        pred_idx, gt_idx = align_gt(pred_idx, gt_idx)

        # append
        agent_traj.append(pred_idx) # agent_traj is 2D array: no. of agents X no. of future timestamps (e.x. 12), pred_idx[0] is a 2D array: no. of future timestamps X 2((x, y) postion) 
        gt_traj.append(np.round(gt_idx,3)) # [vehicles][timesteps,data]

    return np.squeeze(np.array(gt_traj)), np.squeeze(np.array(agent_traj))

'''
Metric for sorting agent locations.
Parameters
    init_positions - ndarray(n_agents, 2) of initial x-y positions of every agent
Returns
    dists_from_origin - ndarray(n_agents, 1) of every agent's initial distance from origin
'''
def sort_metric(init_positions):
    dists_from_origin = np.sqrt(np.square(init_positions[:,0]) + np.square(init_positions[:,1]))
    return dists_from_origin

'''
Aggregate AgentFormer-formatted data for clustering.
Parameters
    dataset_pth - base path of dataset
    split - string indicating train, val, or test split
Returns
    hist - ndarray(n_windows, n_agents, n_history, 3)
    horz - ndarray(n_windows, n_agents, n_horizon, 3)
    crashes - ndarray(n_windows, n_horizon)
'''
def aggregate_data(dataset_pth, split, n_history, n_horizon, max_agents, metric='raw'):
    if split not in ['train', 'val', 'test']:
        raise NotImplementedError
    if metric not in ['raw', 'n_vehicles']:
        raise NotImplementedError
    
    seq_train, seq_val, seq_test = get_racetrack_split_for_fp(dataset_pth)
    seq_eval = locals()[f'seq_{split}']

    hist = []
    horz = []
    crashes = []
    for seq_idx, seq_name in enumerate(seq_eval): # loop through each test episode (i.e., sequence)
        
        # load gt raw data and crash labels from datasets folder
        _, gt_raw, crashes_raw = load_groundtruth_data(dataset_pth, seq_name)

        # grab list of ids and timesteps
        id_list = np.unique(gt_raw[:, 1])
        steps_list = np.unique(gt_raw[:, 0])

        n_agents_missing = max_agents - len(id_list)

        # loop through all windows
        for step_idx in range(n_history-1, len(steps_list)-n_horizon):
            hist_steps = steps_list[step_idx-n_history+1 : step_idx+1]
            horz_steps = steps_list[step_idx+1 : step_idx+n_horizon+1]
            assert len(hist_steps) == n_history and len(horz_steps) == n_horizon

            # loop through each step in this history
            this_hist = []
            for s in hist_steps:
                s_data = gt_raw[gt_raw[:,0] == s][:,[13, 15, 16]]
                s_data = np.pad(s_data, ((0,n_agents_missing),(0,0)), 'constant', constant_values=(200, 200))
                this_hist.append(s_data)
            this_hist = np.swapaxes(np.stack(this_hist), 0, 1).astype(np.float)
            
            # sort agents based on initial distance from origin
            predicate = sort_metric(this_hist[:,0,0:2])
            order = np.argsort(predicate)
            this_hist = this_hist[order,:,:]

            hist.append(this_hist)

            # loop through each step in this horizon
            this_horz = []
            for s in horz_steps:
                s_data = gt_raw[gt_raw[:,0] == s][:,[13, 15, 16]]
                s_data = np.pad(s_data, ((0,n_agents_missing),(0,0)), 'constant', constant_values=(200, 200))
                this_horz.append(s_data)
            this_horz = np.swapaxes(np.stack(this_horz), 0, 1).astype(np.float)

            # sort agents based on initial hist distance from origin
            this_horz = this_horz[order,:,:]

            horz.append(this_horz)

            # add crashes
            if crashes_raw is None:
                crashes.append([None]*n_horizon)
            else:
                crashes.append(crashes_raw[step_idx+1 : step_idx+n_horizon+1])

    hist, horz, crashes = np.stack(hist), np.stack(horz), np.array(crashes)

    return hist, horz, crashes

'''
Caclulate ADE, FDE, minDE, and maxDE
Inputs: ground truth and predicted targets, tensors of size (batch size, num_time_steps, 2)
'''
def calculate_deviation(y, y_pred):
    assert y.shape[0] == y_pred.shape[0]
    N = y.shape[0]
    T = y.shape[1]

    ade, fde, minde, maxde = 0, 0, 0, 0
    for i in range(N):
        errors = []
        for t in range(T):
            errors.append(np.linalg.norm(y[i,t,:] - y_pred[i,t,:], ord=2))
        ade += np.mean(errors)
        fde += errors[-1]
        minde += np.min(errors)
        maxde += np.max(errors)

    return ade/N, fde/N, minde/N, maxde/N

'''
Robust Conformal Prediction
Implementation from https://github.com/SAIDS-Lab/Robust-Conformal-Prediction-for-STL-Runtime-Verification-under-Distribution-Shift/tree/main
'''
def fn(t):
    # We assume to use the total variation distance.
    return 0.5 * abs(t - 1)

def g(epsilon, beta, search_step=0.0007):
    # Check input.
    if beta < 0 or beta > 1:
        raise Exception("Input to the function g is out of range.")

    # Perform a sampling-based line search.
    z = 0
    while z <= 1:
        value = beta * fn(z / beta) + (1 - beta) * fn((1 - z) / (1 - beta))
        if value <= epsilon:
            return z
        z += search_step

    raise Exception("No return from function g.")

def g_inverse(epsilon, tau, search_step=0.0007):
    # Check input.
    if tau < 0 or tau > 1:
        raise Exception("Input to the function g_inverse is out of range.")

    beta = 1
    while beta >= 0:
        if beta != 1 and g(epsilon, beta) <= tau:
            return beta
        beta -= search_step

    raise Exception("No return from function g_inverse.")

def calculate_delta_n(delta, n, epsilon):
    inner = (1 + 1 / n) * g_inverse(epsilon, 1 - delta)
    return (1 - g(epsilon, inner))


def calculate_delta_tilde(delta_n, epsilon):
    answer = 1 - g_inverse(epsilon, 1 - delta_n)
    return answer