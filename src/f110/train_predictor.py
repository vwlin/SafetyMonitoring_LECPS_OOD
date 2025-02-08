# pull data and train network to predict future trajectories
# traces are split into train/val/test traces, which are then saved as frames of (x,y,heading) data

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
import argparse

from utils.predictor import Predictor, TRAIN_HPAMS
from utils.utils import aggregate_data, TrajectoryDataset, calculate_deviation, weighted_mae, eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--finetune',
                        default='None',
                        choices=['None', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0'],
                        help='OOD scenario to finetune.')
    parser.add_argument('--dataset_base', type=str, default='data', help='Base path to dataset if not finetuning.')
    parser.add_argument('--dataset_ft_base', type=str, default='failure_prediction_results', help='Base path to dataset if finetuning.')
    parser.add_argument('--dataset_seed', type=int, default=1, help='Random seed for dataset (trial number).')
    parser.add_argument('--model_base', type=str, default='models', help='Base path of saved non-finetuned model.')
    parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location of saved selected taus.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    args = parser.parse_args()
    print(args)

    if args.finetune == 'None':
        data_base_pth = os.path.join(args.dataset_base, str(args.dataset_seed),  'id')
        model_base_pth = os.path.join(args.model_base, str(args.dataset_seed))
    else:
        assert os.path.exists(args.tau_save_pth), "Tau save path does not exist."
        with open(args.tau_save_pth, 'rb') as f:
            saved_taus = pickle.load(f)

        data_pth = os.path.join(args.dataset_ft_base, str(args.dataset_seed), args.finetune, f'train_{str(saved_taus[args.dataset_seed])}', 'models_acp', 'pred', 'window_data.pkl')
        model_base_pth = os.path.join(f'{args.model_base}_{args.finetune}', str(args.dataset_seed))

        # pull original model and normalization information
        data_stats_load = os.path.join(args.model_base, str(args.dataset_seed), 'normalization.pkl')
        model_load = os.path.join(args.model_base, str(args.dataset_seed), 'predictor.pth')

    n_epochs = TRAIN_HPAMS[args.finetune][0]
    lr = TRAIN_HPAMS[args.finetune][1]
    beta = TRAIN_HPAMS[args.finetune][2]
    weight_decay = 1e-4
    gpu = 0
    batch_size = 64
    eval_freq = 1

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = "cpu"#torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    os.makedirs(model_base_pth, exist_ok=True)

    # load and aggregate train and val data
    if args.finetune == 'None':
        train_data_pth = os.path.join(data_base_pth, 'train.pkl')
        val_data_pth = os.path.join(data_base_pth, 'val.pkl')

        with open(train_data_pth, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_data_pth, 'rb') as f:
            val_data = pickle.load(f)

        X_tr, Y_tr, C_tr = aggregate_data(train_data)
        X_val, Y_val, C_val = aggregate_data(val_data)

        assert X_tr.shape[0] == Y_tr.shape[0] and X_val.shape[0] == Y_val.shape[0]
        n_tr, n_val = X_tr.shape[0], X_val.shape[0]

        # flatten arrays and remove heading from target data
        X_tr, Y_tr = X_tr.reshape(n_tr, -1), Y_tr[:,:,0:2].reshape(n_tr, -1)
        X_val, Y_val = X_val.reshape(n_val, -1), Y_val[:,:,0:2].reshape(n_val, -1)

        # calculate mean and std of X/Y train data for normalization
        X_tr, Y_tr, C_tr = torch.Tensor(X_tr), torch.Tensor(Y_tr), torch.Tensor(C_tr)
        X_val, Y_val, C_val = torch.Tensor(X_val), torch.Tensor(Y_val), torch.Tensor(C_val)

        X_tr_mn, X_tr_std = torch.mean(X_tr,dim=0), torch.std(X_tr,dim=0)
        Y_tr_mn, Y_tr_std = torch.mean(Y_tr,dim=0), torch.std(Y_tr,dim=0)

        train_data_stats = {
            'X_mean' : X_tr_mn,
            'X_std' : X_tr_std,
            'Y_mean': Y_tr_mn,
            'Y_std' : Y_tr_std
        }
        with open(os.path.join(model_base_pth, 'normalization.pkl'), 'wb') as f:
            pickle.dump(train_data_stats, f)

    else:
        # load window data for finetuning
        with open(data_pth, 'rb') as f:
            window_data = pickle.load(f)
        X, Y, C = window_data['X'], window_data['Y'], window_data['crash_windows']
        assert X.shape[0] == Y.shape[0] == C.shape[0]
        p = np.random.permutation(X.shape[0])
        X, Y, C = X[p], Y[p], C[p]

        X_tr, Y_tr, C_tr = X[:int(X.shape[0]*0.9),:,:].reshape(-1,args.n_history*3), Y[:int(X.shape[0]*0.9),:,:].reshape(-1,args.n_horizon*2), C[:int(X.shape[0]*0.9)]
        X_val, Y_val, C_val = X[int(X.shape[0]*0.9):,:,:].reshape(-1,args.n_history*3), Y[int(X.shape[0]*0.9):,:,:].reshape(-1,args.n_horizon*2), C[int(X.shape[0]*0.9):]
        assert X_tr.shape[0] == Y_tr.shape[0] == C_tr.shape[0] and X_val.shape[0] == Y_val.shape[0] == C_val.shape[0]

        X_tr, Y_tr, C_tr = torch.Tensor(X_tr), torch.Tensor(Y_tr), torch.Tensor(C_tr)
        X_val, Y_val, C_val = torch.Tensor(X_val), torch.Tensor(Y_val), torch.Tensor(C_val)

        # load train data stats from original model
        with open(data_stats_load, 'rb') as f:
            train_data_stats = pickle.load(f)
        
        X_tr_mn = train_data_stats['X_mean']
        X_tr_std = train_data_stats['X_std']
        Y_tr_mn = train_data_stats['Y_mean']
        Y_tr_std = train_data_stats['Y_std']

    # generate datasets and dataloaders
    train_dataset = TrajectoryDataset(X_tr, Y_tr, C_tr, (X_tr_mn, X_tr_std), (Y_tr_mn, Y_tr_std))
    val_dataset = TrajectoryDataset(X_val, Y_val, C_val, (X_tr_mn, X_tr_std), (Y_tr_mn, Y_tr_std))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    n_train_b = len(train_dataloader)

    # train network
    model = Predictor().to(device)
    if args.finetune != 'None':
        checkpoint = torch.load(model_load)
        model.load_state_dict(checkpoint['model_state_dict'])
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr_mn, X_tr_std, Y_tr_mn, Y_tr_std = X_tr_mn.to(device), X_tr_std.to(device), Y_tr_mn.to(device), Y_tr_std.to(device)

    # init eval
    eval_mse, eval_mae, eval_wmae, eval_ade, eval_fde, eval_minde, eval_maxde = eval_model(model, val_dataloader, device, args.n_horizon, beta, Y_tr_mn, Y_tr_std)
    print('epoch %s - eval MSE/MAE/wMAE(%.3f, %.3f, %.3f), eval deviations: (%.3f, %.3f, %.3f, %.3f)\n'
            % (-1, eval_mse, eval_mae, eval_wmae, eval_ade, eval_fde, eval_minde, eval_maxde))

    model.train()
    best_loss = float("Inf")
    tr_losses, val_losses = [], []
    for epoch in range(n_epochs):
        tr_mse, tr_mae, tr_wmae, tr_ade, tr_fde, tr_minde, tr_maxde = 0, 0, 0, 0, 0, 0, 0
        for x, y, c in train_dataloader:
            x, y, c = x.to(device), y.to(device), c.to(device)

            y_pred = model(x)
            if args.finetune == 'None':
                loss = weighted_mae(y_pred, y, c, beta)
            else:
                loss = weighted_mae(y_pred, y, c, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training information
            tr_mse, tr_mae, tr_wmae = tr_mse+loss, tr_mae+mae(y_pred, y), tr_wmae+weighted_mae(y_pred, y, c, beta)
            y_og = (y*Y_tr_std+Y_tr_mn).reshape(-1,args.n_horizon,2).detach().cpu().numpy()
            y_pred_og = (y_pred*Y_tr_std+Y_tr_mn).reshape(-1,args.n_horizon,2).detach().cpu().numpy()
            ade, fde, minde, maxde = calculate_deviation(y_og, y_pred_og)
            tr_ade, tr_fde, tr_minde, tr_maxde = tr_ade+ade, tr_fde+fde, tr_minde+minde, tr_maxde+maxde
        
        tr_mse, tr_mae, tr_wmae, tr_ade, tr_fde, tr_minde, tr_maxde = tr_mse/n_train_b, tr_mae/n_train_b, tr_wmae/n_train_b, tr_ade/n_train_b, tr_fde/n_train_b, tr_minde/n_train_b, tr_maxde/n_train_b
        tr_losses.append([tr_mse.detach().cpu().numpy(), tr_mae.detach().cpu().numpy(), tr_wmae.detach().cpu().numpy(),
                        tr_ade, tr_fde, tr_minde, tr_maxde])

        # eval and save if best model so far
        if epoch % eval_freq == 0:
            eval_mse, eval_mae, eval_wmae, eval_ade, eval_fde, eval_minde, eval_maxde = eval_model(model, val_dataloader, device, args.n_horizon, beta, Y_tr_mn, Y_tr_std)
            val_losses.append([eval_mse.detach().cpu().numpy(), eval_mae.detach().cpu().numpy(), eval_wmae.detach().cpu().numpy(),
                            eval_ade, eval_fde, eval_minde, eval_maxde])
            print('epoch %s - train MSE/MAE/wMAE: (%.3f, %.3f, %.3f), train deviations: (%.3f, %.3f, %.3f, %.3f), eval MSE/MAE/wMAE: (%.3f, %.3f, %.3f), eval deviations: (%.3f, %.3f, %.3f, %.3f)\n'
                % (epoch, tr_mse, tr_mae, tr_wmae, tr_ade, tr_fde, tr_minde, tr_maxde, eval_mse, eval_mae, eval_wmae, eval_ade, eval_fde, eval_minde, eval_maxde))

            if args.finetune == 'None':
                keep_criteria = eval_wmae
            else:
                # keep_criteria = eval_mae
                keep_criteria = eval_wmae
            if keep_criteria < best_loss:
                best_loss = keep_criteria
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tr_loss': (tr_mse, tr_mae, tr_wmae),
                    'eval_loss': (eval_mse, eval_mae, eval_wmae),
                    'tr_deviations':(tr_ade, tr_fde, tr_minde, tr_maxde),
                    'eval_deviations':(eval_ade, eval_fde, eval_minde, eval_maxde)
                }, os.path.join(model_base_pth, 'predictor.pth'))
                print(f'saved at epoch {epoch}\n')

    print('best result - average eval loss %.5f' % best_loss)

    tr_losses, val_losses = np.array(tr_losses), np.array(val_losses)
    epoch_list = list(np.arange(0,n_epochs,eval_freq))

    _, ax = plt.subplots(3,1)
    for i in range(3):
        ax[i].plot(tr_losses[:,i], marker='.', label='train')
        ax[i].plot(epoch_list, val_losses[:,i], marker='.', label='eval')
        ax[i].set_xlabel('epoch')
    ax[0].set_ylabel('MSE')
    ax[1].set_ylabel('MAE')
    ax[2].set_ylabel('wMAE')
    plt.legend()
    plt.savefig(os.path.join(model_base_pth, 'mse_mae.png'))
    plt.show()

    _, ax = plt.subplots(2,2)
    ct = 3
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(tr_losses[:,ct], marker='.', label='train')
            ax[i,j].plot(epoch_list, val_losses[:,ct], marker='.', label='eval')
            ax[i,j].set_xlabel('epoch')
            ct += 1
    ax[0,0].set_ylabel('ADE')
    ax[0,1].set_ylabel('FDE')
    ax[1,0].set_ylabel('MinDE')
    ax[1,1].set_ylabel('MaxDE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_base_pth, 'deviations.png'))
    plt.show()