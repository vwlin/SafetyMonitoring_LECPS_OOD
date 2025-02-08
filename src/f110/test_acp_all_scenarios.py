'''
Test is all on original model (non-finetuned)
'''
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Data split to use.')

    # predictions
    parser.add_argument('--gt_pred', default='pred', choices=['gt','pred'], help='Evaluate on groundtruth or prediction data.')
    parser.add_argument('--predictions_base', type=str, default='models', help='Base path to model predictions.')
    
    # failure prediction parameters
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
    
    # incremental learning
    parser.add_argument('--tau_save_pth', type=str, default='logs/select_ncs_threshold/taus.pkl', help='Location of saved selected taus.')
    parser.add_argument('--use_incremental', action='store_true', help='Use arg to use incremental learning.')
    parser.add_argument('--cluster_method', default='kmeans', choices=['mems','kmeans'], help='Cluster method.')
    
    args = parser.parse_args()

    print(args)

    n_history, n_horizon = 5, 5
    start_t = (n_history+2*n_horizon-1)

    fig, ax = plt.subplots(2,1, figsize=(4,5), sharex=True)

    x_axis_range = 0
    for scenario in ['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0']:
        if args.use_incremental and scenario == 'id':
            continue

        max_t = 201
        longest_timesteps = list(range(1,max_t))

        # begin gathering data to plot results of empirical tests of Lemma 1 and Theorem 1
        coverage_emp_prob = [[] for _ in range(max_t)]
        stl_satisfied_emp_prob = [[] for _ in range(max_t)]
        for dataset_seed in range(1,11):
            if args.use_incremental:
                predictions_base = f'{args.predictions_base}_{scenario}'
            else:
                predictions_base = args.predictions_base

            # get results directory
            fp_results_dir = make_fp_results_dir(args.fp_results_root, dataset_seed, scenario, args.split, predictions_base, args.gt_pred,
                                                    args.use_incremental, args.cluster_method,
                                                    'acp', None)
            # os.makedirs(os.path.join(fp_results_dir, 'acp_figures'), exist_ok=True)

            # load acp info
            with open(os.path.join(fp_results_dir, 'acp_data.pkl'), 'rb') as f:
                acp_info = pickle.load(f)

            gamma = acp_info.pop('gamma')
            delta = acp_info.pop('delta')

            episodes = list(acp_info.keys()) # aka sequences
            # print(f'Gamma: {gamma}, Delta: {delta}, {len(episodes)} episodes')

            ep_lens = []
            for e in episodes:
                far = acp_info[e].pop('far')
                mar = acp_info[e].pop('mar')
                fp = acp_info[e].pop('false_pos')
                tn = acp_info[e].pop('true_neg')
                fn = acp_info[e].pop('false_neg')
                tp = acp_info[e].pop('true_pos')
                steps_metric = acp_info[e].pop('steps')

                timesteps = list(acp_info[e].keys())
                ep_lens.append(len(timesteps))

                # print(f'Episode {e} with {len(timesteps)} steps: FAR {far}, MAR {mar} No. Steps {steps_metric}')

                ''' for empirical test of Lemma 1 '''
                rhos = np.array([acp_info[e][t]['rho'] for t in timesteps])
                rho_hats = np.array([acp_info[e][t]['rho_hat'] for t in timesteps])
                C_ts = np.array([acp_info[e][t]['C_t'] for t in timesteps])
                
                ncs = rho_hats - rhos
                coverage = (ncs <= C_ts)

                for t in range(len(coverage)):
                    coverage_emp_prob[t].append(coverage[t])

                ''' for empirical test of Theorem 1 '''
                # get time until which thm 1's condition holds
                thm_1_cond = (rho_hats >= C_ts)
                false_locations = np.where(thm_1_cond == False)[0]
                if len(false_locations) > 0:
                    stop_idx = false_locations[0]
                else:
                    stop_idx = len(timesteps)
                # print(f'Episode {e}: condition for theorem 1 is satisfied until step {stop_idx}')

                # for time in [0, stop_idx), check whether STL specification is satisfied
                stl_satisfied = (rhos[:stop_idx] > 0)
                # print(stl_satisfied)
                for t in range(stop_idx):
                    stl_satisfied_emp_prob[t].append(stl_satisfied[t])

        # to ensure there are enough data points to get a good estimate of empirical coverage at each time t, only gather data up until t < stop_pt
        stop_pt = int(np.round(np.mean(ep_lens)))
        if x_axis_range < stop_pt:
            x_axis_range = stop_pt

        label = {'id':'ID', 'ood_0.0_3':'3 rays', 'ood_0.0_5':'5 rays', 'ood_0.9_0':'noise (0.9)', 'ood_1.0_0':'noise (1.0)'}[scenario]
        # color = {'id':'black', 'ood_0.0_3':cm.tab10(0), 'ood_0.0_5':cm.tab10(1), 'ood_0.9_0':cm.tab10(2), 'ood_1.0_0':cm.tab10(3)}[scenario]
        
        ''' for Lemma 1 '''
        # average across all traces in all seeds
        coverage_emp_prob = [np.mean(coverage_emp_prob[t]) for t in range(stop_pt) if len(coverage_emp_prob[t])>0]

        # get ave over time as time increases, and plot for each scenario
        expanding_ave = np.array([np.mean(coverage_emp_prob[:i]) for i in range(1, len(coverage_emp_prob)+1)])
        ax[0].plot(list(range(start_t, len(expanding_ave)+start_t)), expanding_ave, alpha=0.7)#, color=color)

        ''' for Theorem 1 '''
        # average across all traces in all seeds
        stl_satisfied_emp_prob = [np.mean(stl_satisfied_emp_prob[t]) for t in range(stop_pt) if len(stl_satisfied_emp_prob[t])>0]

        # get ave over time as time increases, and plot for each scenario
        expanding_ave = np.array([np.mean(stl_satisfied_emp_prob[:i]) for i in range(1, len(stl_satisfied_emp_prob)+1)])
        ax[1].plot(list(range(start_t, len(expanding_ave)+start_t)), expanding_ave, alpha=0.7, label=label)#, color=color)

    # get envelopes for acp convergence rate
    p1 = (delta+gamma)/(np.array(longest_timesteps)*gamma)
    p2 = (1-delta+gamma)/(np.array(longest_timesteps)*gamma)

    d = .015

    ##############################
    ##############################

    ''' for Lemma 1 '''
    # plot bound lines and use broken axes when scale necessitates
    # https://stackoverflow.com/questions/44731152/matplotlib-create-broken-axis-in-subplot
    # https://stackoverflow.com/questions/17976103/broken-axis-example-uneven-subplot-size
    ax[0].axhline(y = 1-delta, color='black', linestyle='--', linewidth=1, label='1-$\delta$')
    ax[0].plot(np.array(longest_timesteps)+start_t, 1-delta-p1, color='grey', linestyle='--', linewidth=1, label='theoretical bounds')
    
    ratio = 0.3
    lem1_divider = make_axes_locatable(ax[0])
    lem1_bounds = lem1_divider.new_vertical(size=f"{ratio*100}%", pad=0.1, pack_start=False)
    fig.add_axes(lem1_bounds)
    lem1_bounds.plot(np.array(longest_timesteps)+start_t, 1-delta+p2, color='grey', linestyle='--', linewidth=1)
    lem1_bounds.set_ylim(1.0, 4.0)
    lem1_bounds.tick_params(bottom=False, labelbottom=False)
    lem1_bounds.spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    kwargs = dict(transform=lem1_bounds.transAxes, color='k', clip_on=False)
    lem1_bounds.plot((-d, +d), (-d/ratio, +d/ratio), **kwargs)        # top-left diagonal
    lem1_bounds.plot((1 - d, 1 + d), (-d/ratio, +d/ratio), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax[0].transAxes)  # switch to the bottom axes
    ax[0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # set labelings, etc
    ax[0].set_xlim([0,x_axis_range+start_t+20])
    ax[0].set_ylim([0.1,0.95])
    lem1_bounds.set_ylabel('Empirical Coverage Rate')
    lem1_bounds.yaxis.set_label_coords(0.04,.8, transform=fig.transFigure)
    lem1_bounds.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ''' for Theorem 1 '''
    # plot bound lines and use broken axes when scale necessitates
    # https://stackoverflow.com/questions/44731152/matplotlib-create-broken-axis-in-subplot
    # https://stackoverflow.com/questions/17976103/broken-axis-example-uneven-subplot-size
    ax[1].axhline(y = 1, color='grey', linestyle='--', linewidth=1)
    ax[1].set_ylim([0.97, 1.005])
    ax[1].set_yticks([0.98, 0.99, 1.0])

    ratio = 0.4
    thm1_divider = make_axes_locatable(ax[1])
    thm1_bounds = thm1_divider.new_vertical(size=f"{ratio*100}%", pad=0.1, pack_start=True)
    fig.add_axes(thm1_bounds)
    thm1_bounds.plot(np.array(longest_timesteps)+start_t, 1-delta-p1, color='grey', linestyle='--', linewidth=1)
    thm1_bounds.set_ylim([0.4, 0.9])
    ax[1].tick_params(bottom=False, labelbottom=False)
    ax[1].spines['bottom'].set_visible(False)
    thm1_bounds.spines['top'].set_visible(False)
    kwargs = dict(transform=ax[1].transAxes, color='k', clip_on=False)
    ax[1].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=thm1_bounds.transAxes)  # switch to the bottom axes
    thm1_bounds.plot((-d, +d), (1 - d/ratio, 1 + d/ratio), **kwargs)  # bottom-left diagonal
    thm1_bounds.plot((1 - d, 1 + d), (1 - d/ratio, 1 + d/ratio), **kwargs)  # bottom-right diagonal

    # set labelings, etc
    thm1_bounds.set_xlabel('Length of Trace')
    thm1_bounds.set_ylabel('Empirical Satisfaction Rate')
    thm1_bounds.yaxis.set_label_coords(0.04,0.3, transform=fig.transFigure)
    thm1_bounds.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # ax[1].legend(lines, labels, prop={'size': 9})
    ax[1].legend(prop={'size': 9})

    ##############################
    ##############################

    # ''' for Lemma 1 '''
    # # ax[0].axhline(y = 1-delta, color='grey', linestyle='--', linewidth=1)
    # # ax[0].plot(np.array(longest_timesteps)+start_t, 1-delta-p1, color='grey', linestyle='--', linewidth=1)
    # # ax[0].plot(np.array(longest_timesteps)+start_t, 1-delta+p2, color='grey', linestyle='--', linewidth=1)

    # ax[0].set_xlim([0,x_axis_range+start_t+20])
    # # ax[0].set_ylim([-2,10])
    # ax[0].set_ylabel('Empirical Coverage Rate')

    # ''' for Theorem 1 '''
    # # ax[1].axhline(y = 1, color='grey', linestyle='--', linewidth=1)
    # # ax[1].plot(np.array(longest_timesteps)+start_t, 1-delta-p1, color='grey', linestyle='--', linewidth=1)

    # ax[1].set_xlabel('Length of Trace')
    # ax[1].set_ylabel('Empirical Satisfaction Rate')
    # # ax[1].set_ylim([-2,2])
    # # ax[1].legend()#prop={'size': 9})

    ##############################
    ##############################

    ''' save '''
    plt.tight_layout()
    os.makedirs(os.path.join(args.fp_results_root, 'acp_figures'), exist_ok=True)
    plt.savefig(os.path.join(args.fp_results_root, 'acp_figures', f'thm1_lmm1_all_scenarios_{args.use_incremental}.png'))
    plt.close()