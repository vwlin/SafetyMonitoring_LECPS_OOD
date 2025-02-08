import numpy as np
import pickle
import os
import argparse

from utils.utils import calculate_deviation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--finetune',
                        default='None',
                        choices=['None', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='OOD scenario to finetune. Use None to evaluate the non-finetuned model.')
    parser.add_argument('--model_base', type=str, default='models', help='Base path of saved non-finetuned model.')
    parser.add_argument('--dataset_base', type=str, default='data', help='Base path to dataset.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--n_seeds', type=int, default=10, help='Number of seeds to evaluate.')
    parser.add_argument('--memories_location', default=None, help='Location of memories. If none supplied, default location will be used.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Split of data to evaluate on.')
    args = parser.parse_args()
    print(args)

    if args.finetune == 'None':
        scenarios = ['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0']
    else:
        scenarios = [args.finetune]

    for scenario in scenarios:
        print(f'Scenario: {scenario}')
        ade, fde, minde, maxde = [], [], [], []
        for seed in range(1,1+args.n_seeds):             
            # load predictions
            f0_model_base_pth = os.path.join(args.model_base, str(seed))
            f0_predictions_pth = os.path.join(f0_model_base_pth, 'eval', scenario)
            with open(os.path.join(f0_predictions_pth, f'{args.split}_predictions.pkl'), 'rb') as f:
                f0_predictions = pickle.load(f)

            if args.finetune != 'None':
                f1_model_base_pth = os.path.join(f'{args.model_base}_{args.finetune}', str(seed))
                f1_predictions_pth = os.path.join(f1_model_base_pth, 'eval', scenario)
                with open(os.path.join(f1_predictions_pth, f'{args.split}_predictions.pkl'), 'rb') as f:
                    f1_predictions = pickle.load(f)
                assert f1_predictions.keys() == f0_predictions.keys()

                # load memories
                memory_pth = os.path.join(f1_model_base_pth, 'pseudo_memories.pkl') if args.memories_location is None else args.memories_location
                with open(memory_pth, 'rb') as f:
                    pseudo_memories = pickle.load(f)
                kmeans = pseudo_memories['kmeans']
                ood_clusters = pseudo_memories['ood_clusters']

            # aggregate gt / prediction pairs
            traces = f0_predictions.keys()
            y, y_pred = [], []
            for trace in traces:
                f0_y = f0_predictions[trace]['Y']
                f0_y_pred = f0_predictions[trace]['Y_pred']

                if args.finetune == 'None':
                    y.append(f0_y)
                    y_pred.append(f0_y_pred)
                else:
                    x = f0_predictions[trace]['X']
                    assert np.all(x == f1_predictions[trace]['X'])

                    f1_y = f1_predictions[trace]['Y']
                    f1_y_pred = f1_predictions[trace]['Y_pred']

                    cluster = kmeans.predict(x.reshape(-1, args.n_history*3).astype('float'))
                    is_ood = np.array([c in ood_clusters for c in cluster])

                    y.append(f0_y[~is_ood])
                    y.append(f1_y[is_ood])
                    y_pred.append(f0_y_pred[~is_ood])
                    y_pred.append(f1_y_pred[is_ood])

            y = np.concatenate(y)
            y_pred = np.concatenate(y_pred)

            # calculate metrics
            _ade, _fde, _minde, _maxde = calculate_deviation(y, y_pred)
            ade.append(_ade)
            fde.append(_fde)
            minde.append(_minde)
            maxde.append(_maxde)

        print(f'ADE: {np.mean(ade):.4f} / {np.std(ade):.4f}, FDE: {np.mean(fde):.4f} / {np.std(fde):.4f}, MinDE: {np.mean(minde):.4f} / {np.std(minde):.4f}, MaxDE: {np.mean(maxde):.4f} / {np.std(maxde):.4f}\n')
