from multiprocessing.spawn import import_main_path
import gym
from tqdm import tqdm
from time import sleep
import numpy as np
from os.path import dirname, join
from os import makedirs
from gym.wrappers import TimeLimit, OrderEnforcing
from gym.wrappers.env_checker import PassiveEnvChecker
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import torch
import argparse
import math

from dqn_agent import DQN_Agent

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ood_type',
                    default='gravity',
                    choices=['gravity', 'cart_mass', 'pole_length', 'pole_mass','force_magnitude'],
                    help='Type of OOD scenario.')
args = parser.parse_args()
print(args)

# ood_vals
ood_g = [0.98, 1.09, 1.23, 1.4, 1.63, 1.96, 2.45, 3.27, 4.9, 9.8,
         19.6, 29.4, 39.2, 49.0, 58.8, 68.6, 78.4, 88.2, 98.0]
id_g = 9.8

ood_m_cart = [0.1, 0.1111, 0.125, 0.1429, 0.1667, 0.2, 0.25, 0.3333,
              0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
id_m_cart = 1.0

ood_l = [0.05, 0.0556, 0.0625, 0.0714, 0.0833, 0.1, 0.125,
         0.1667, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
id_l = 0.5

ood_m_pole = [0.01, 0.0111, 0.0125, 0.0143, 0.0167, 0.02, 0.025, 0.0333,
              0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
id_m_pole = 0.1

ood_f = [1.0, 1.1111, 1.25, 1.4286, 1.6667, 2.0, 2.5, 3.3333, 5.0,
         10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
id_f = 10.0

ood_vals = {'gravity' : (id_g, ood_g),
            'cart_mass' : (id_m_cart, ood_m_cart),
            'pole_length' : (id_l, ood_l),
            'pole_mass' : (id_m_pole, ood_m_pole),
            'force_magnitude' : (id_f, ood_f)}

horizon = 20
episodes = 100
d = 5 # 4d state, 1d action
seed = 0

current_dir = dirname(__file__)
chkpt = join(current_dir, "cartpole-dqn.pth")
makedirs('figs', exist_ok=True)

# ---------- Helper Functions ---------- #
def gather_data(env, agent, episodes, horizon, d, stl_angle_threshold=12*2*math.pi/360, stl_x_threshold=2.4):
    cumulative_rewards = []
    robustness_values = []
    trajectories = np.zeros((episodes, horizon*d))
    for i in tqdm(range(episodes)):
        terminated, truncated = False, False
        obs, _ = env.reset()

        ct = 0
        running_reward = 0
        this_episode_violations = []
        while not terminated and not truncated:
            A = agent.get_action(obs, env.action_space.n, epsilon=0)
            action = A.item()

            obs, reward, terminated, truncated, _ = env.step(action)

            if ct < horizon:
                this_episode_violations.append(min( -np.abs(obs[2])+stl_angle_threshold, -np.abs(obs[0])+stl_x_threshold ))
                trajectories[i, d*ct:d*ct+d] = [obs[0], obs[1], obs[2], obs[3], action]
                ct += 1
            else:
                break
                
            running_reward += reward

        robustness_values.append(np.min(this_episode_violations))
        cumulative_rewards.append(running_reward)

    ave_cumulative_reward =  np.mean(cumulative_rewards)
    ave_robustness_value = np.mean(robustness_values) # robustness value for STL formula G[0,horizon)(pole_angle <= stl_angle_threshold and cart_x <= stl_x_threshold), averaged over all episodes

    return ave_cumulative_reward, ave_robustness_value, trajectories

def test_on_agent(agent):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = gym.make('CartPole-v1')
    env.action_space.seed(seed)

    # ---------- Gather ID Statistics ---------- #
    id_ave_reward, id_ave_rob_val, id_trajectories = gather_data(env, agent, episodes, horizon, d)

    id_density = KernelDensity(kernel='gaussian').fit(id_trajectories)

    # ---------- Run tests ---------- #
    ood_ave_rewards = []
    ood_ave_rob_vals = []
    Ls = [] # likelihoods

    for new_val in ood_vals[args.ood_type][1]:
        print('Testing', args.ood_type, ':', new_val)
        # get ood samples
        env = gym.make('CartPole-v1')
        env.action_space.seed(seed)
        env = env.unwrapped
        
        if args.ood_type == 'gravity':
            env.gravity = new_val
        elif args.ood_type == 'cart_mass':
            env.masscart = new_val
        elif args.ood_type == 'pole_length':
            env.length = new_val
        elif args.ood_type == 'pole_mass':
            env.masspole = new_val
        elif args.ood_type == 'force_magnitude':
            env.force_mag = new_val
        else:
            raise NotImplementedError

        env = PassiveEnvChecker(env)
        env = OrderEnforcing(env)
        env = TimeLimit(env, 500)

        ood_ave_reward, ood_ave_rob_val, ood_trajectories = gather_data(env, agent, episodes, horizon, d)

        ood_ave_rewards.append(ood_ave_reward)

        ood_ave_rob_vals.append(ood_ave_rob_val) # robustness value for STL formula G[0,9](pole_angle <= 12), averaged over all episodes
        
        scores = id_density.score_samples(ood_trajectories)
        scores = np.exp(scores)
        ood_Ls = np.sum(scores)/episodes
        Ls.append(ood_Ls)

    '''
    Plot as three superimposed.
    https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
    '''
    x = ood_vals[args.ood_type][1]
    fig, host = plt.subplots(figsize=(3,2))
    ax2 = host.twinx()
    ax3 = host.twinx()

    # limits
    host.set_ylim(-5,horizon+5)
    ax2.set_ylim(-0.08,0.20)
    ax3.set_ylim([-0.1e-42,2.0e-42])

    # labels
    host.set_xlabel(args.ood_type)
    host.set_ylabel("Reward")
    ax2.set_ylabel("Robustness Value")
    ax3.set_ylabel("Likelihood")

    # colors
    color1, color2, color3 = 'tab:blue','tab:orange','tab:green'

    # data
    host.axvline(ood_vals[args.ood_type][0], color='k', linestyle='--', label=f'ID {args.ood_type}')
    p1 = host.plot(x, ood_ave_rewards, color=color1, label="Reward")
    p2 = ax2.plot(x, ood_ave_rob_vals, color=color2, label="Robustness Value")
    p3 = ax3.plot(x, Ls, color=color3, label="Likelihood")

    # axes locations
    ax3.spines['right'].set_position(('outward', 50))
    ax3.yaxis.offsetText.set_position((1.8, 0))

    # label colors
    host.yaxis.label.set_color(color1)
    ax2.yaxis.label.set_color(color2)
    ax3.yaxis.label.set_color(color3)

    # axis tick colors
    host.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    ax3.tick_params(axis='y', colors=color3)

    plt.savefig(f"figs/cartpole_{args.ood_type}_all.png", bbox_inches='tight')

    '''
    Uncomment the blocks below to generate variants of the above plot without
    the left/right axes. This is useful for creating figures for publication.
    '''
    # host.set_yticks([])
    # host.set_ylabel('')
    # plt.savefig(f"figs/cartpole_{args.ood_type}_all_noleft.png", bbox_inches='tight')

    # host.yaxis.set_major_locator(AutoLocator())
    # host.set_ylabel("Reward")
    # ax2.set_yticks([])
    # ax2.set_ylabel('')
    # ax3.set_yticks([])
    # ax3.set_ylabel('')
    # plt.savefig(f"figs/cartpole_{args.ood_type}_all_noright.png", bbox_inches='tight')

    # host.set_yticks([])
    # host.set_ylabel('')
    # plt.savefig(f"figs/cartpole_{args.ood_type}_all_none.png", bbox_inches='tight')


if __name__ == '__main__':
    # env info for agent
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    exp_replay_size = 256

    # trained agent
    trained_agent = DQN_Agent(seed=0, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                    exp_replay_size=exp_replay_size)
    trained_agent.load_pretrained_model(chkpt)
    test_on_agent(trained_agent)