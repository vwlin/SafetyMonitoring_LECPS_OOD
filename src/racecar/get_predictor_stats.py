import numpy as np
import pickle
import os
import argparse

from utils.utils import *
from AgentFormer.utils.utils import find_unique_common_from_lists, load_list_from_folder, load_txt_file

def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.mean(axis=-1)                       # samples
        ade += dist.min(axis=0)                         # (1, )
    ade /= len(pred_arr)
    return ade


def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples 
        fde += dist.min(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde

def compute_MaxDE(pred_arr, gt_arr):
    maxde = 0.0
    for pred, gt in zip(pred_arr, gt_arr): # loop through agents
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.max(axis=-1)                        # samples
        maxde += dist.min(axis=0)                       # (1, )
    maxde /= len(pred_arr)
    return maxde

def compute_MinDE(pred_arr, gt_arr):
    minde = 0.0
    for pred, gt in zip(pred_arr, gt_arr): # loop through agents
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist.min(axis=-1)                        # samples
        minde += dist.min(axis=0)                       # (1, )
    minde /= len(pred_arr)
    return minde


def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--finetune',
                        default='None',
                        choices=['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5'],
                        help='OOD scenario to finetune.')
    parser.add_argument('--model_base', type=str, default='AgentFormer', help='Base path of saved non-finetuned model.')
    parser.add_argument('--dataset_base', type=str, default='AgentFormer/datasets/racetrack', help='Base path to dataset.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of seeds to evaluate.')
    parser.add_argument('--memories_location', default=None, help='Location of memories. If none supplied, default location will be used.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Split of data to evaluate on.')
    args = parser.parse_args()
    print(args)

    if args.finetune == 'None':
        scenarios = ['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5']
    else:
        scenarios = [args.finetune]

    for scenario in scenarios:
        n_agents = {'id':2,'vehicles_2':3,'vehicles_3':4,'vehicles_4':5,'vehicles_5':6}[scenario]
        ade, fde, minde, maxde = [], [], [], []
        for seed in range(args.n_seeds):
            print(f'Scenario: {scenario}, seed: {seed}')
            seeded_model_base = os.path.join(args.model_base, f'results_s{seed}')

            # update predicitons_base argument based on best epoch
            metrics_file = os.path.join(seeded_model_base, 'racetrack_agentformer_pre', 'validation_metrics.pkl') # f0
            with open(metrics_file, 'rb') as f:
                metrics = pickle.load(f)
            best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
            if scenario == 'id':
                predictions_base = os.path.join(seeded_model_base, f'racetrack_agentformer_pre/results/epoch_{best_epoch:04d}/{args.split}/recon')
            else:
                predictions_base = os.path.join(seeded_model_base, f'racetrack_{scenario}_ogtest/results/epoch_{best_epoch:04d}/{args.split}/recon')

            if args.finetune != 'None':
                f1_metrics_file = os.path.join(seeded_model_base, f'racetrack_{scenario}_fttest', 'validation_metrics.pkl')
                with open(f1_metrics_file, 'rb') as f:
                    f1_metrics = pickle.load(f)
                f1_best_epoch = min(f1_metrics, key=lambda x: f1_metrics[x]['ADE'])
                f1_predictions_base = os.path.join(seeded_model_base, f'racetrack_{scenario}_fttest/results/epoch_{f1_best_epoch:04d}/{args.split}/recon')

                # load memories
                memory_pth = os.path.join(seeded_model_base, f'racetrack_{scenario}_fttest/pseudo_memories.pkl') if args.memories_location is None else args.memories_location
                with open(memory_pth, 'rb') as f:
                    pseudo_memories = pickle.load(f)
                kmeans = pseudo_memories['kmeans']
                ood_clusters = pseudo_memories['ood_clusters']

            # prepare to load data
            seq_train, seq_val, seq_test = get_racetrack_split_for_fp(f'{args.dataset_base}/{scenario}_s{seed}')
            seq_eval = globals()[f'seq_{args.split}']

            # get dict of ego vehicle IDs
            with open(os.path.join(args.dataset_base, f'{scenario}_s{seed}', f'ego_ids_{args.split}.pkl'), 'rb') as f:
                ego_ids_dict = pickle.load(f)
            seed_ade, seed_fde, seed_minde, seed_maxde = [] ,[] ,[] ,[]
            for seq_name in seq_eval:
                print(seq_name)

                # load gt raw data and crash labels from datasets folder
                gt_raw, gt_raw_all, crashes = load_groundtruth_data(f'{args.dataset_base}/{scenario}_s{seed}', seq_name)
                data_filelist, _ = load_list_from_folder(os.path.join(predictions_base, seq_name))
                if args.finetune != 'None':
                    f1_data_filelist, _ = load_list_from_folder(os.path.join(f1_predictions_base, seq_name))
                    assert len(data_filelist) == len(f1_data_filelist)

                for (d,data_file) in enumerate(data_filelist):
                    # load reconstructed data from AgentFormer, and process reconstructed and groundtruth data
                    ego_id = float(ego_ids_dict[int(seq_name.split('_')[1])])
                    horz_gt, horz_pred = process_reconstructed_and_gt_data(data_file, gt_raw, ego_id)
            
                    if args.finetune == 'None':
                        predicted_traj = horz_pred
                    else:
                        # if using incremental learning, load reconstructed data from finetuned AgentFormer
                        f1_horz_gt, f1_horz_pred = process_reconstructed_and_gt_data(f1_data_filelist[d], gt_raw, ego_id)
                        assert (f1_horz_gt == horz_gt).all()

                        # grab history for clustering
                        current_timestep = int(os.path.basename(data_file).split('_')[1].split('.')[0])
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
                            predicted_traj = horz_pred
                        else:
                            predicted_traj = f1_horz_pred

                    seed_ade.append(compute_ADE(predicted_traj, horz_gt))
                    seed_fde.append(compute_FDE(predicted_traj, horz_gt))
                    seed_minde.append(compute_MinDE(predicted_traj, horz_gt))
                    seed_maxde.append(compute_MaxDE(predicted_traj, horz_gt))

            ade.append(np.mean(seed_ade))
            fde.append(np.mean(seed_fde))
            minde.append(np.mean(seed_minde))
            maxde.append(np.mean(seed_maxde))

        print(f'ADE: {np.mean(ade):.4f} / {np.std(ade):.4f}, FDE: {np.mean(fde):.4f} / {np.std(fde):.4f}, MinDE: {np.mean(minde):.4f} / {np.std(minde):.4f}, MaxDE: {np.mean(maxde):.4f} / {np.std(maxde):.4f}\n')
