import numpy as np
from torch.utils.data import Dataset
import math
import torch
import os

from utils.f110_env import WIDTH, LENGTH, VERT_WALL_Xs, HORZ_WALL_Ys

# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class TrajectoryDataset(Dataset):
    """TensorDataset with support of transforms.
    X_stats, Y_stats are tuples of (mean, std) of X, Y data
    C is boolean crash labels for horizons
    """
    def __init__(self, X, Y, C, X_stats=None, Y_stats=None):
        assert X.shape[0] == Y.shape[0] == C.shape[0]
        self.X = X
        self.Y = Y
        self.C = C
        self.X_stats = X_stats
        self.Y_stats = Y_stats

    def __getitem__(self, index):
        x = self.X[index]

        if self.X_stats:
            x = (x - self.X_stats[0]) / self.X_stats[1]

        y = self.Y[index]

        if self.Y_stats:
            y = (y - self.Y_stats[0]) / self.Y_stats[1]

        c = self.C[index]

        return x, y, c

    def __len__(self):
        return self.X.shape[0]
    
def aggregate_data(data_dict):
    traces = list(data_dict.keys())
    X = data_dict[traces[0]]['X']
    Y = data_dict[traces[0]]['Y']
    crashes = data_dict[traces[0]]['crash_windows']
    for trace in traces[1:]:
        X = np.concatenate((X, data_dict[trace]['X']))
        Y = np.concatenate((Y, data_dict[trace]['Y']))
        crashes = np.concatenate((crashes, data_dict[trace]['crash_windows'])).astype(bool)
    return X, Y, crashes

def weighted_mae(output, target, crashes, beta):
    loss = torch.abs(output - target)

    safe, crash = loss[crashes == 0], loss[crashes == 1]
    assert len(safe) > 0 or len(crash) > 0

    loss_safe = torch.mean(safe)
    loss_crash = torch.mean(crash)
    if len(safe) > 0 and len(crash) > 0:
        loss = beta*loss_safe + (1-beta)*loss_crash
    elif len(safe) > 0:
        loss = loss_safe
    else: #len(crash > 0)
        loss = loss_crash
    return loss

'''
Evaluate model
Parameters
    model
    dataloader
    device
    n_horizon - length of prediction horizon
    beta - weighted MAE hyperparameter
    Y_tr_mn - mean of Y training data
    X_tr_std - standard deviation of Y training data
Returns
    MSE, MAE, weighted MAE, ADE, FDE, MinDE, MaxDE
'''
def eval_model(model, dataloader, device, n_horizon, beta, Y_tr_mn, Y_tr_std):
    n = len(dataloader)

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    with torch.no_grad():
        model.eval()
        mse, mae, wmae, ade, fde, minde, maxde = 0, 0, 0, 0, 0, 0, 0
        for x, y, c in dataloader:
            x, y, c = x.to(device), y.to(device), c.to(device)

            y_pred = model(x)

            mse, mae, wmae = mse+mse_loss(y_pred, y), mae+mae_loss(y_pred, y), wmae+weighted_mae(y_pred, y, c, beta)

            y_denormed = (y*Y_tr_std+Y_tr_mn).reshape(-1,n_horizon,2).detach().cpu().numpy()
            y_pred_denormed = (y_pred*Y_tr_std+Y_tr_mn).reshape(-1,n_horizon,2).detach().cpu().numpy()
            ade_, fde_, minde_, maxde_ = calculate_deviation(y_denormed, y_pred_denormed)
            ade, fde, minde, maxde = ade+ade_, fde+fde_, minde+minde_, maxde+maxde_

        mse, mae, wmae, ade, fde, minde, maxde = mse/n, mae/n, wmae/n, ade/n, fde/n, minde/n, maxde/n
        model.train()

    return mse, mae, wmae, ade, fde, minde, maxde

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
assuming safety property to be no collision with wall, safety_threshold is the min
distance that we want the ego vehicle to maintain from all the walls,
computes robustness at each future timestamp and returns min value of
robustess among these future timestamps
Parameters
    trajectories - ndarray(n_horizon,2)
'''
def compute_robustness(trajectories, safety_threshold):    
    min_dist = np.Inf
    
    for i in range(trajectories.shape[0]): # iterating over future timesteps
        
        # calculate robustness value and track min over time

        # CORNERS
        # bottom left
        # plt.scatter(-1.5/2, -10, c='r')
        # plt.scatter(1.5/2, -10+1.5, c='r')
        # top left
        # plt.scatter(-1.5/2, 10, c='r')
        # plt.scatter(1.5/2, 10-1.5, c='r')
        # top right
        # plt.scatter(20-1.5/2, 10, c='r')
        # plt.scatter(20-4.5/2, 10-1.5, c='r')
        # bottom right
        # plt.scatter(20-1.5/2, -10, c='r')
        # plt.scatter(20-4.5/2, -10+1.5, c='r')

        dists = []
        # external vertical walls
        dists.append(np.abs(VERT_WALL_Xs[0] - trajectories[i,0]))
        dists.append(np.abs(VERT_WALL_Xs[3] - trajectories[i,0]))
        # external horizontal walls
        dists.append(np.abs(HORZ_WALL_Ys[0] - trajectories[i,1]))
        dists.append(np.abs(HORZ_WALL_Ys[3] - trajectories[i,1]))
        # internal vertical walls
        if trajectories[i,1] > -LENGTH/2+WIDTH-safety_threshold and trajectories[i,1] < LENGTH/2-WIDTH+safety_threshold:
            dists.append(np.abs(VERT_WALL_Xs[1] - trajectories[i,0]))
            dists.append(np.abs(VERT_WALL_Xs[2] - trajectories[i,0]))
        # internal horizontal walls
        if trajectories[i,0] > WIDTH/2-safety_threshold and trajectories[i,0] < LENGTH-WIDTH-WIDTH/2+safety_threshold:
            dists.append(np.abs(HORZ_WALL_Ys[1] - trajectories[i,1]))
            dists.append(np.abs(HORZ_WALL_Ys[2] - trajectories[i,1]))

        i_min = np.min(dists)
        if i_min < min_dist:
            min_dist = i_min     
        
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
                        tau=None, epsilon=0.08):

    fp_results_dir = os.path.join(fp_results_root, str(dataset_seed), scenario, split)
    if tau != None:
        fp_results_dir = f'{fp_results_dir}_{str(tau)}'

    fp_results_dir = os.path.join(fp_results_dir, predictions_base)
    if use_incremental:
        fp_results_dir = f'{fp_results_dir}_{cluster_method}'
        if cluster_method == 'mems':
            fp_results_dir = f'{fp_results_dir}-{cluster_metric}'
        fp_results_dir = f'{fp_results_dir}-il'

    if pred_method == 'point': pred_method = 'pp'
    fp_results_dir = f'{fp_results_dir}_{pred_method}'
    if pred_method == 'rcp':
        fp_results_dir = f'{fp_results_dir}_{epsilon}'

    fp_results_dir = os.path.join(fp_results_dir, gt_pred)
    os.makedirs(fp_results_dir, exist_ok=True)

    return fp_results_dir

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