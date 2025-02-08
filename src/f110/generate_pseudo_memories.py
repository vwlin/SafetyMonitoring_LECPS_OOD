import pickle
import argparse
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os

from utils.utils import aggregate_data
from utils.f110_env import plot_halls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--finetune',
                        default='ood_0.0_3',
                        choices=['ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='OOD scenario to finetune.')
    parser.add_argument('--dataset_base', type=str, default='data', help='Base path to dataset.')
    parser.add_argument('--dataset_ft_base', type=str, default='failure_prediction_results', help='Base path to dataset for finetuning.')
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    parser.add_argument('--model_base', type=str, default='models', help='Base path of saved non-finetuned model.')
    parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location of saved selected taus.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    args = parser.parse_args()
    print(args)

    threshold = {
        'ood_0.0_3':0.25,
        'ood_0.0_5':0.25,
        'ood_0.9_0':0.05,
        'ood_1.0_0':0.05
    }[args.finetune]

    # import ID data
    data_pth = os.path.join(args.dataset_base, str(args.dataset_seed), 'id/train.pkl')

    with open(data_pth, 'rb') as f:
        train_data = pickle.load(f)
    id_hist, _, _ = aggregate_data(train_data)

    # load taus
    assert os.path.exists(args.tau_save_pth), "Tau save path does not exist."
    with open(args.tau_save_pth, 'rb') as f:
        saved_taus = pickle.load(f)

    # import OOD data
    data_pth = os.path.join(args.dataset_ft_base, str(args.dataset_seed), args.finetune, f'train_{str(saved_taus[args.dataset_seed])}', f'{args.model_base}_acp', 'pred/window_data.pkl')

    with open(data_pth, 'rb') as f:
        window_data = pickle.load(f)
    ood_hist = window_data['X'].astype('float')

    # concatenate data
    hist = np.concatenate((id_hist, ood_hist)).reshape(-1,args.n_history*3)

    # select k
    k_vals = range(5,20)
    sse = []
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(hist)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(k_vals, sse, curve="convex", direction="decreasing")
    k = kl.elbow
    print('best k:', k)

    plt.plot(list(k_vals), sse)
    plt.axvline(k, linestyle='--')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.savefig(os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed), 'kmeans_elbow.png'))
    plt.close()

    # do clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(hist)
    print(kmeans.cluster_centers_.reshape(-1,args.n_history,3))

    # do predictions
    id_predictions = kmeans.predict(id_hist.reshape(-1,args.n_history*3))
    ood_predictions = kmeans.predict(ood_hist.reshape(-1,args.n_history*3))
    id_unique, id_counts = np.unique(id_predictions, return_counts=True)
    ood_unique, ood_counts = np.unique(ood_predictions, return_counts=True)
    id_counts = id_counts / np.sum(id_counts)
    ood_counts = ood_counts / np.sum(ood_counts)
    print('id:', id_unique, id_counts)
    print('ood:', ood_unique, ood_counts)
    plt.hist(id_predictions)
    plt.savefig(os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed), 'kmeans_id_predictions.png'))
    plt.close()
    plt.hist(ood_predictions)
    plt.savefig(os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed), 'kmeans_ood_predictions.png'))
    plt.close()

    mask = np.where(ood_counts > threshold)
    ood_clusters = ood_unique[mask]
    print('ood clusters:', ood_clusters)

    # plot clusters
    for (c,cluster_center) in enumerate(kmeans.cluster_centers_):
        if c in ood_clusters:
            cluster_center = cluster_center.reshape(args.n_history,3)
            plt.plot(cluster_center[:,0], cluster_center[:,1], label=c) #, color=color[c])
    plot_halls()
    plt.legend()
    plt.savefig(os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed), 'kmeans_clusters.png'))
    plt.close()

    pseudo_mems = {
        'kmeans':kmeans,
        'ood_clusters':ood_clusters
    }
    with open(os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed), 'pseudo_memories.pkl'), 'wb') as f:
        pickle.dump(pseudo_mems, f)