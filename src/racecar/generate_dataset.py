import os
import json
import pickle
import random
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw_base', type=str, default='../../raw_data/racecar_data', help='Base path to raw data.')
    parser.add_argument('--dest_base', type=str, default='AgentFormer/datasets/racetrack', help='Base path to store generated dataset.')
    parser.add_argument('--n_horizon', type=int, default=5, help='Length of prediction horizon.')
    parser.add_argument('--n_history', type=int, default=5, help='Length of history input.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()
    print(args)

    # handle seeding, define dataset splits, and make directories if necessary
    random.seed(args.seed)
    id_split = [0.65, 0.15, 0.2]
    ood_split = [0.5, 0, 0.5] # "train" - data without IL, "test" - data with IL
    assert np.sum(id_split)==1.0 and np.sum(ood_split)==1.0, 'Splits must sum to 1'

    # perform data generation for each scenario
    for scenario in ['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5']:
        print(f'\nProcessing data for scenario {scenario}')

        dest_dir = os.path.join(args.dest_base, f'{scenario}_s{str(args.seed)}')
        assert not os.path.exists(dest_dir), ' '.join([dest_dir, 'directory already exists. Delete directory, or dataset generation result will be undefined.'])
        os.makedirs(dest_dir, exist_ok=False)

        if 'vehicles' in scenario:
            split = ood_split
        else:
            split = id_split

        # read raw data
        with open(os.path.join(args.raw_base, f'{scenario}.json')) as f:
            raw_data = json.loads(f.read()) # episode -> timestep -> 'crashed' or ID -> [x, y, heading]
        episode_list = list(raw_data.keys())# episodes aka sequences

        # randomly split episodes into train/val/test episodes
        random.shuffle(episode_list)
        n_traces = len(episode_list)
        n_train = int(n_traces*split[0])
        n_val = int(n_traces*split[1])
        n_test = n_traces - n_val - n_train
        train_episodes = episode_list[:n_train]
        val_episodes = episode_list[n_train:n_train+n_val]
        test_episodes = episode_list[n_train+n_val:]
        assert len(train_episodes) == n_train and len(val_episodes) == n_val and len(test_episodes) == n_test

        # loop through all scenarios
        filenum = 0
        for mode in ['train', 'val', 'test']:
            postfix = mode
            if mode == 'train':
                mode_episodes = train_episodes
            elif mode == 'val':
                mode_episodes = val_episodes
            else:
                mode_episodes = test_episodes
                postfix = ''

            # save a text file for each episode (this ensures that sliding window in AgentFormer code does not contain data from two different episodes)
            crashes_count = [False]*len(mode_episodes)
            mapping = []
            ego_id_dict = {}
            for (e, episode) in enumerate(mode_episodes):
                episode_data = raw_data[episode] # timestep -> 'crashed' or ID -> [x, y, heading]
                
                ego_id = episode_data.pop('ego_id')
                steps = list(episode_data.keys())

                if len(steps) < args.n_history + args.n_horizon:
                    print(f'Cannot construct a full frame from {scenario} episode {str(episode)}. Skipping.')
                    continue

                data_str = []
                crash_flags, crash_windows = [], []
                for step in steps:
                    step_data = episode_data[step] # 'crashed' or ID -> [x, y, heading]
                    
                    crash_label = step_data['crashed']
                    crash_flags.append(crash_label)
                    crashes_count[e] = crashes_count[e] or crash_label

                    agents = list(step_data.keys())
                    agents.remove('crashed')

                    for agent in agents:
                        agent_data = [
                            step,                                       # timestep
                            agent,                                      # id
                            'Car',                                      # agent class
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,     # unused quantities
                            step_data[agent][0],                        # x (x by AgentFormer naming conventions)
                            -1,                                         # unused quantity
                            step_data[agent][1],                        # y (z by Agentformer naming conventions)
                            step_data[agent][2]                         # heading
                        ]
                        data_str.append(' '.join([str(d) for d in agent_data]))

                    if int(step) >= args.n_history and int(step) < len(steps)-args.n_horizon+1:
                        crash_windows.append(np.any(np.array([episode_data[str(s)]['crashed'] for s in range(int(step), int(step)+args.n_horizon)])))
                crash_windows = np.array(crash_windows, dtype=bool)

                # save data in file format required by AgentFormer
                data_file_str = '\n'.join(data_str)
                data_filename = '_'.join(filter(None,['racetrack',str(filenum),postfix])) + '.txt'
                with open(os.path.join(dest_dir, data_filename), 'w') as f:
                    f.write(data_file_str)
                
                # save label data in file format required by Failure Prediction algorithm
                label_data = {
                    'crash_flags':list(crash_flags),
                    'crash_windows':list(crash_windows)
                }
                label_filename = '_'.join(filter(None,['racetrack',str(filenum),'labels',postfix])) + '.pkl'
                with open(os.path.join(dest_dir, label_filename), 'wb') as f:
                    pickle.dump(label_data, f)

                # record ego IDs
                ego_id_dict[filenum] = ego_id
                with open(os.path.join(dest_dir, f'ego_ids_{mode}.pkl'), 'wb') as f:
                    pickle.dump(ego_id_dict, f)

                # save episode to file number mapping
                mapping.append(f'{str(episode)} {str(filenum)}')
                filenum += 1

            print(f'crashes in {mode} set:', np.sum(crashes_count))

        mapping_str = f'seed {(args.seed)}\n'
        mapping_str += '\n'.join(mapping)
        with open(os.path.join(dest_dir, 'seed_filenum_mapping.txt'), 'w') as f:
            f.write(mapping_str)