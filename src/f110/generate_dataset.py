# read raw lidar data and split into train/val/test data
# traces are split into train/val/test traces, which are then saved as frames of (x,y,heading) data

import os
import json
import random
import numpy as np
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw_base', type=str, default='../../raw_data/f110_data', help='Base path to raw data.')
    parser.add_argument('--dest_base', type=str, default='data', help='Base path to store generated dataset.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()
    print(args)

    scenarios = {'new_train_data':'id',
                 'ood_data_0.0_3':'ood_0.0_3',
                 'ood_data_0.0_5':'ood_0.0_5',
                 'ood_data_0.9_0':'ood_0.9_0',
                 'ood_data_1.0_0':'ood_1.0_0'}

    random.seed(args.seed)
    id_split = [0.65, 0.15, 0.2]
    ood_split = [0.5, 0, 0.5] # "train" - data without IL, "test" - data with IL
    assert np.sum(id_split)==1.0 and np.sum(ood_split)==1.0, 'Splits must sum to 1'
    for scenario in scenarios:
        print(f'\nProcessing data for scenario {scenario}')

        dest_dir = os.path.join(args.dest_base, str(args.seed), scenarios[scenario])
        os.makedirs(dest_dir, exist_ok=True)

        if 'ood' in scenario:
            split = ood_split
        else:
            split = id_split

        raw_dir = f'{args.raw_base}/{scenario}'
        raw_pths = os.listdir(raw_dir)
        random.shuffle(raw_pths)

        n_traces = len(raw_pths)
        n_train = int(n_traces*split[0])
        n_val = int(n_traces*split[1])
        n_test = n_traces - n_val - n_train
        
        train_data, val_data, test_data = {}, {}, {}
        total_frames = [0, 0, 0]
        crash_times = {'train':{}, 'val':{}, 'test':{}}
        for (i,raw_pth) in enumerate(raw_pths):
            if ('json' not in raw_pth) or ('setting' in raw_pth):
                continue

            with open(f'{raw_dir}/{raw_pth}') as f:
                trace = json.load(f)
            steps = np.sort(np.array(list(trace.keys()), dtype=int))

            if len(steps) < args.n_history + args.n_horizon:
                print(f'Cannot construct a full frame from {raw_pth}. Skipping.')
                continue

            X, Y = [], []
            crash_flags = []
            crash_windows = []
            for step in steps:
                if step >= args.n_history and step < len(steps)-args.n_horizon+1:
                    history = np.array([[trace[str(s)][0], trace[str(s)][2], trace[str(s)][3]] for s in range(step-args.n_history, step)])
                    horizon = np.array([[trace[str(s)][0], trace[str(s)][2], trace[str(s)][3]] for s in range(step, step+args.n_horizon)])
                    crash_windows.append(np.any(np.array([trace[str(s)][-1] for s in range(step, step+args.n_horizon)])))
                    X.append(history)
                    Y.append(horizon)
                crash_flags.append(trace[str(step)][-1])
            X, Y = np.stack(X), np.stack(Y)
            crash_windows = np.array(crash_windows, dtype=bool)
            if True in crash_flags:
                crash_time = crash_flags.index(True)
            else:
                crash_time = -1
            crash_flags = np.array(crash_flags)

            trace_dict = {
                'X':X,
                'Y':Y,
                'crash_labels':crash_flags, # whether there is a crash at each time step
                'crash_windows':crash_windows # whether there is a crash in each window
            }
            if i < n_train:
                train_data[raw_pth] = trace_dict
                total_frames[0] += X.shape[0]
                mode = 'train'
            elif i < n_train + n_val:
                val_data[raw_pth] = trace_dict
                total_frames[1] += X.shape[0]
                mode = 'val'
            else:
                test_data[raw_pth] = trace_dict
                total_frames[2] += X.shape[0]
                mode = 'test'

            if crash_time in crash_times[mode]:
                crash_times[mode][crash_time] += 1
            else:
                crash_times[mode][crash_time] = 1
        # end for
        
        if len(train_data.keys()) > 0:
            with open(os.path.join(dest_dir, 'train.pkl'), 'wb') as f:
                pickle.dump(train_data, f)
        if len(val_data.keys()) > 0:
            with open(os.path.join(dest_dir, 'val.pkl'), 'wb') as f:
                pickle.dump(val_data, f)
        if len(test_data.keys()) > 0:
            with open(os.path.join(dest_dir, 'test.pkl'), 'wb') as f:
                pickle.dump(test_data, f)
        
        print(f'{total_frames[0]} train frames, {total_frames[1]} val frames, and {total_frames[2]} test frames saved.')
        
        for mode in crash_times.keys():
            sorted_keys = list(crash_times[mode].keys())
            sorted_keys.sort()
            crash_times[mode] = {i: crash_times[mode][i] for i in sorted_keys}
            print(f'{mode} crash times:')
            print(crash_times[mode])
    # end for