import argparse
import pickle
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os

from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario',
                        default='vehicles_2',
                        choices=['vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5'],
                        help='OOD scenario to generate clusters for.')
    parser.add_argument('--dataset_base', type=str, default='AgentFormer/datasets/racetrack', help='Base path to dataset.')
    parser.add_argument('--dataset_seed', type=int, default=0, help='Random seed for dataset (trial number).')
    parser.add_argument('--model_base', type=str, default='AgentFormer/results', help='Base path of saved non-finetuned model.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    args = parser.parse_args()
    print(args)

    threshold = {
        'vehicles_2':0.1,
        'vehicles_3':0.1,
        'vehicles_4':0.1,
        'vehicles_5':0.1
    }[args.scenario]

    save_dir = os.path.join(f'{args.model_base}_s{args.dataset_seed}', f'racetrack_{args.scenario}_fttest')
    n_agents = int(args.scenario.split('_')[1]) + 1

    print('loading data...')
    # import ID train data
    id_dataset_pth = os.path.join(args.dataset_base, f'id_s{str(args.dataset_seed)}')
    id_hist, _, _ = aggregate_data(id_dataset_pth, 'train', args.n_history, args.n_horizon, n_agents)

    # import OOD train data (high NCS windows)
    ood_dataset_pth = os.path.join(args.dataset_base, f'{args.scenario}_windows_s{str(args.dataset_seed)}')
    ood_hist, _, _ = aggregate_data(ood_dataset_pth, 'train', args.n_history, args.n_horizon, n_agents)

    # concatenate data
    hist = np.concatenate((id_hist, ood_hist)).reshape(-1,n_agents*args.n_history*3)

    # select k
    print('selecting k...')
    k_vals = range(5,20)
    sse = []
    for k in k_vals:
        print(f'testing k={str(k)}...')
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(hist)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(k_vals, sse, curve="convex", direction="decreasing")
    k = kl.elbow
    print('best k:', k)

    plt.plot(list(k_vals), sse)
    plt.axvline(k, linestyle='--')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.savefig(os.path.join(save_dir, 'kmeans_elbow.png'))
    plt.close()

    # do clustering
    print('clustering...')
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(hist)

    # do predictions
    print('predicting...')
    id_predictions = kmeans.predict(id_hist.reshape(-1,n_agents*args.n_history*3))
    ood_predictions = kmeans.predict(ood_hist.reshape(-1,n_agents*args.n_history*3))
    id_unique, id_counts = np.unique(id_predictions, return_counts=True)
    ood_unique, ood_counts = np.unique(ood_predictions, return_counts=True)
    id_counts = id_counts / np.sum(id_counts)
    ood_counts = ood_counts / np.sum(ood_counts)
    print('id:', id_unique, id_counts)
    print('ood:', ood_unique, ood_counts)
    plt.hist(id_predictions)
    plt.savefig(os.path.join(save_dir, 'kmeans_id_predictions.png'))
    plt.close()
    plt.hist(ood_predictions)
    plt.savefig(os.path.join(save_dir, 'kmeans_ood_predictions.png'))
    plt.close()

    mask = np.where(ood_counts > threshold)
    ood_clusters = ood_unique[mask]
    print('ood clusters:', ood_clusters)

    # plot clusters
    fig, ax = plt.subplots(n_agents, 1, figsize=(5,3*n_agents))
    fig.suptitle('Cluster Centers')
    for a_i, a in enumerate(ax):
        a.set_xlabel('x position')
        a.set_ylabel('y position')
        a.title.set_text(f'Agent {a_i}')

    for (c,cluster_center) in enumerate(kmeans.cluster_centers_):
        if c in ood_clusters:
            cluster_center = cluster_center.reshape(n_agents, args.n_history,3)
            for a in range(n_agents):
                ax[a].plot(cluster_center[a,:,0], cluster_center[a,:,1], label=c) #, color=color[c])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kmeans_clusters.png'))
    plt.close()

    pseudo_mems = {
        'kmeans':kmeans,
        'ood_clusters':ood_clusters
    }
    with open(os.path.join(save_dir, 'pseudo_memories.pkl'), 'wb') as f:
        pickle.dump(pseudo_mems, f)
