import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from utils.predictor import Predictor, TRAIN_HPAMS
from utils.utils import aggregate_data, TrajectoryDataset, eval_model
from utils.f110_env import plot_halls

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--finetune',
                        default='None',
                        choices=['None', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='OOD scenario to finetune.')
    parser.add_argument('--model_base', type=str, default='models', help='Base path of saved non-finetuned model.')
    parser.add_argument('--dataset_base', type=str, default='data', help='Base path to dataset.')
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    args = parser.parse_args()
    print(args)

    if args.finetune == 'None':
        model_base_pth = os.path.join(args.model_base, str(args.dataset_seed))
        scenarios = ['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0']
    else:
        model_base_pth = os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed))
        scenarios = [args.finetune]
        
    batch_size = 256
    device = "cpu"#torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    # load model checkpoint
    checkpoint = torch.load(os.path.join(model_base_pth, 'predictor.pth'))
    model = Predictor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load stats for normalization
    with open(os.path.join(args.model_base, str(args.dataset_seed), 'normalization.pkl'), 'rb') as f:
        normalize_stats = pickle.load(f)
    X_mn = normalize_stats['X_mean']
    X_std = normalize_stats['X_std']
    Y_mn = normalize_stats['Y_mean']
    Y_std = normalize_stats['Y_std']

    # metrics
    for scenario in scenarios:
        save_base_pth = os.path.join(model_base_pth, 'eval', scenario)

        for mode in ['train', 'val', 'test']:
            if mode == 'val' and scenario != 'id':
                continue
            print(scenario, mode)

            # load and aggregate test data
            data_pth = os.path.join(args.dataset_base, str(args.dataset_seed), scenario, f'{mode}.pkl')
            with open(data_pth, 'rb') as f:
                test_data = pickle.load(f)

            X_agg, Y_agg, C_agg = aggregate_data(test_data)
            assert X_agg.shape[0] == Y_agg.shape[0] == C_agg.shape[0]
            n_te = X_agg.shape[0]

            # flatten arrays and remove heading from target data
            X_agg, Y_agg = X_agg.reshape(n_te, -1), Y_agg[:,:,0:2].reshape(n_te, -1)

            X_agg, Y_agg, C_agg = torch.Tensor(X_agg), torch.Tensor(Y_agg), torch.Tensor(C_agg)

            # generate and dataloader
            test_dataset = TrajectoryDataset(X_agg, Y_agg, C_agg, (X_mn, X_std), (Y_mn, Y_std))
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # get test metrics
            ave_mse, ave_mae, ave_wmae, ave_ade, ave_fde, ave_minde, ave_maxde = eval_model(model, test_dataloader, device, args.n_horizon, TRAIN_HPAMS[args.finetune][2], Y_mn, Y_std)
            print('MSE: %.3f, MAE: %.3f, wMAE: %.3f, ADE: %.3f, FDE: %.3f, MinDE: %.3f, MaxDE: %.3f'
                % (ave_mse, ave_mae, ave_wmae, ave_ade, ave_fde, ave_minde, ave_maxde))

            # get trace-wise predictions
            fig_dir = os.path.join(save_base_pth, f'{mode}_figs')
            os.makedirs(fig_dir, exist_ok=True)
            predictions = {}
            for (t,trace) in enumerate(test_data.keys()):
                X = test_data[trace]['X']
                Y = test_data[trace]['Y']
                assert X.shape[0] == Y.shape[0]

                X = torch.Tensor(X.reshape(X.shape[0], -1))
                Y =  torch.Tensor(Y[:,:,0:2].reshape(Y.shape[0], -1))

                X = (X - X_mn) / X_std
                Y = (Y - Y_mn) / Y_std

                Y_pred = model(X)

                X_denormed = (X * X_std + X_mn).reshape(-1,args.n_history,3).detach().numpy()
                Y_denormed = (Y * Y_std + Y_mn).reshape(-1,args.n_horizon,2).detach().numpy()
                Y_pred_denormed = (Y_pred * Y_std + Y_mn).reshape(-1,args.n_horizon,2).detach().numpy()
                predictions[trace] = {'X':X_denormed,
                                    'Y':Y_denormed,
                                    'Y_pred':Y_pred_denormed}
                
                # plot ground truth
                plot_halls()
                gt_plot = X_denormed[0,:,0:2].squeeze()
                for i in range(X_denormed.shape[0]):
                    gt_plot = np.concatenate((gt_plot, X_denormed[i,-1,0:2].reshape(1,2)))
                gt_plot = np.concatenate((gt_plot, Y_denormed[-1,:,:].squeeze()))
                plt.plot(gt_plot[:,0], gt_plot[:,1], c='b', label='ground truth')
                for i in range(Y_pred_denormed.shape[0]):
                    if i == 0:
                        plt.plot(Y_pred_denormed[i,:,0], Y_pred_denormed[i,:,1], '--', c='r', label='predictions')
                    else:
                        plt.plot(Y_pred_denormed[i,:,0], Y_pred_denormed[i,:,1], '--', c='r')
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                if 'ood' in scenario:
                    plt.title(f'{scenario} {trace} ({Y_pred_denormed.shape[0]} steps)')
                else:
                    plt.title(f'iD {trace} ({Y_pred_denormed.shape[0]} steps)')
                plt.savefig(os.path.join(fig_dir, f'{trace}.png'))
                plt.close() 

            with open(os.path.join(save_base_pth, f'{mode}_predictions.pkl'), 'wb') as f:
                pickle.dump(predictions, f)