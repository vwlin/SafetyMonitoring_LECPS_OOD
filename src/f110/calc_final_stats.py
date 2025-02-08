import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
import os

from utils.plot import categorical_cmap

SCENARIOS = ['id', 'ood_0.0_3', 'ood_0.0_5', 'ood_0.9_0', 'ood_1.0_0']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--summary_results_pth', type=str, default='results_summary.pkl', help='Location of results summary.')
    parser.add_argument('--case', choices=['crash', 'safe', 'all'], default='all', help='Calculate metrics across crash cases or safe cases.')    
    parser.add_argument('--save_dir', type=str, default='figs', help='Directory to save figures.')
    parser.add_argument('--epsilon', type=float, default=0.08, help='RCP epsilon')    
    args = parser.parse_args()

    print(args)

    with open(args.summary_results_pth, 'rb') as f:
        all_results = pickle.load(f)

    print(all_results.keys())

    os.makedirs(args.save_dir, exist_ok=True)

    '''
    Aggregate results
    '''
    masked_results = {}
    stats = {'point_False':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'cp_False':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'rcp_False':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'acp_False':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'point_True':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'cp_True':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'rcp_True':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]},
             'acp_True':{'far_mn':[],'far_std':[], 'mar_mn':[], 'mar_std':[], 'prec_mn':[], 'prec_std':[], 'rec_mn':[], 'rec_std':[], 'steps_mn':[], 'steps_std':[]}}
    test_incomplete = False
    for scenario in SCENARIOS:
        for use_incremental in [False, True]:
            if scenario == 'id' and use_incremental == True:
                continue

            for use_acp in ['point', 'cp', 'rcp', 'acp']:
                if use_acp == 'rcp':
                    key = f'{scenario}_{use_incremental}_{use_acp}_{args.epsilon}'
                else:
                    key = f'{scenario}_{use_incremental}_{use_acp}'
                if key not in all_results:
                    test_incomplete = True
                    continue
                these_results = all_results[key]
                masked_results[key] = {'far_list':[], 'mar_list':[], 'precision_list':[], 'recall_list':[], 'steps_list':[]}

                for seed in range(1,11):
                    far = these_results['far_list'][seed]
                    mar = these_results['mar_list'][seed]
                    precision = these_results['precision_list'][seed]
                    recall = these_results['recall_list'][seed]
                    steps = these_results['steps_list'][seed]

                    # get idxs of crash and safe cases. if there are none, skip this seed and don't calculate any metrics
                    if args.case == 'crash':
                        idx = np.argwhere(~np.isnan(mar))
                        if len(idx) == 0: continue
                        far, mar = np.array(far)[idx], np.array(mar)[idx]
                    elif args.case == 'safe':
                        idx = np.argwhere(np.isnan(mar))
                        if len(idx) == 0: continue
                        far, mar = np.array(far)[idx], np.array(mar)[idx]

                    masked_results[key]['far_list'].append(np.nanmean(far)*100)
                    masked_results[key]['precision_list'].append(np.nanmean(precision))
                    masked_results[key]['recall_list'].append(np.nanmean(recall))
                    if args.case != 'safe':
                        masked_results[key]['mar_list'].append(np.nanmean(mar)*100)
                        masked_results[key]['steps_list'].append(np.nanmean(steps))

                all_seeds_far = masked_results[key]['far_list']
                all_seeds_prec = masked_results[key]['precision_list']
                all_seeds_rec = masked_results[key]['recall_list']
                if args.case != 'safe':
                    all_seeds_mar = masked_results[key]['mar_list']
                    all_seeds_steps = masked_results[key]['steps_list']

                new_key = f'{str(use_acp)}_{str(use_incremental)}'
                stats[new_key]['far_mn'].append(np.nanmean(all_seeds_far))
                stats[new_key]['far_std'].append(np.nanstd(all_seeds_far))
                stats[new_key]['prec_mn'].append(np.nanmean(all_seeds_prec))
                stats[new_key]['prec_std'].append(np.nanstd(all_seeds_prec))
                stats[new_key]['rec_mn'].append(np.nanmean(all_seeds_rec))
                stats[new_key]['rec_std'].append(np.nanstd(all_seeds_rec))
                if args.case != 'safe':
                    stats[new_key]['mar_mn'].append(np.nanmean(all_seeds_mar))
                    stats[new_key]['mar_std'].append(np.nanstd(all_seeds_mar))
                    stats[new_key]['steps_mn'].append(np.nanmean(all_seeds_steps))
                    stats[new_key]['steps_std'].append(np.nanstd(all_seeds_steps))

                print(scenario, use_incremental, use_acp)
                print(f"FAR: {stats[new_key]['far_mn'][-1]:.2f} / {stats[new_key]['far_std'][-1]:.2f}")
                if args.case != 'safe':
                    print(f"MAR: {stats[new_key]['mar_mn'][-1]:.2f} / {stats[new_key]['mar_std'][-1]:.2f}")
                    print(f"Steps: {stats[new_key]['steps_mn'][-1]:.2f} / {stats[new_key]['steps_std'][-1]:.2f}")
                print(f"Precision: {stats[new_key]['prec_mn'][-1]:.2f} / {stats[new_key]['prec_std'][-1]:.2f}")
                print(f"Recall: {stats[new_key]['rec_mn'][-1]:.2f} / {stats[new_key]['rec_std'][-1]:.2f}")
                print()

    if not test_incomplete:
        '''
        Calculate statistical significances
        '''
        print('''Running significance tests...
            Reject null hypothesis of identical means if p sufficiently low.
            ''')
        if args.case != 'safe':
            metric_list = ['far', 'mar', 'steps', 'precision', 'recall']
        else:
            metric_list = ['far']

        print('Significance of OOD')
        for scenario in SCENARIOS[1:]:
            for metric in metric_list:
                a = masked_results[f'id_False_point'][f'{metric}_list']
                b = masked_results[f'{scenario}_False_point'][f'{metric}_list']
                t, p = ttest(a, b, equal_var=False)

                print(f'ID vs {scenario}, {metric}: {p}')
        print()

        print('Significance of adding ACP without IL') # (ID, OOD_0.0_3, OOD_0.0_5, OOD_0.9_0, OOD_1.0_0)
        for scenario in SCENARIOS:
            for metric in metric_list:
                a = masked_results[f'{scenario}_False_point'][f'{metric}_list']
                b = masked_results[f'{scenario}_False_acp'][f'{metric}_list']
                t, p = ttest(a, b, equal_var=False)

                print(f'{scenario}, {metric}: {p}')
        print()

        print('Significance of adding IL without ACP') # (ID, OOD_0.0_3, OOD_0.0_5, OOD_0.9_0, OOD_1.0_0)
        for scenario in SCENARIOS[1:]:
            for metric in metric_list:
                a = masked_results[f'{scenario}_False_point'][f'{metric}_list']
                b = masked_results[f'{scenario}_True_point'][f'{metric}_list']
                t, p = ttest(a, b, equal_var=False)

                print(f'{scenario}, {metric}: {p}')
        print()

        print('Significance of adding IL with ACP') # (ID, OOD_0.0_3, OOD_0.0_5, OOD_0.9_0, OOD_1.0_0)
        for scenario in SCENARIOS[1:]:
            for metric in metric_list:
                a = masked_results[f'{scenario}_False_acp'][f'{metric}_list']
                b = masked_results[f'{scenario}_True_acp'][f'{metric}_list']
                t, p = ttest(a, b)

                print(f'{scenario}, {metric}: {p}')

        '''
        Plot results
        '''
        # results with rcp
        tight = False
        if tight:
            X_axis = np.arange(len(SCENARIOS))*4.0 # 2.8
            bar_width = 0.3 # 0.2
            space_btwn_bars = 0.22 # 0.15
            cluster_center = space_btwn_bars*4
        else:
            X_axis = np.arange(len(SCENARIOS))*5.5
            bar_width = 0.4
            space_btwn_bars = 0.3
        cluster_center = space_btwn_bars*4
        colors = categorical_cmap(2,4,cmap="tab10").colors
        ebar_size = 2
        ebar_color = 'black'

        for stat in ['far', 'mar', 'steps', 'prec', 'rec']:
            if args.case == 'safe' and stat in ['mar', 'steps']:
                continue

            stat_name = {
                'far':'False Alarm Rate (%)',
                'mar':'Missed Alarm Rate (%)',
                'steps':'Steps',
                'prec':'Precision',
                'rec':'Recall'
            }[stat]

            if not tight: plt.figure(figsize=(8,4))

            x_pp_nil = [X_axis[i]-(cluster_center+3*space_btwn_bars)    if i > 0 else X_axis[i]-3*space_btwn_bars for i in range(len(X_axis))]
            x_cp_nil = [X_axis[i]-(cluster_center+space_btwn_bars)      if i > 0 else X_axis[i]-space_btwn_bars for i in range(len(X_axis))]
            x_rcp_nil = [X_axis[i]-(cluster_center-space_btwn_bars)     if i > 0 else X_axis[i]+space_btwn_bars for i in range(len(X_axis))]
            x_acp_nil = [X_axis[i]-(cluster_center-3*space_btwn_bars)   if i > 0 else X_axis[i]+3*space_btwn_bars for i in range(len(X_axis))]
            x_pp_wil = X_axis[1:]+(cluster_center-3*space_btwn_bars)
            x_cp_wil = X_axis[1:]+(cluster_center-space_btwn_bars)
            x_rcp_wil = X_axis[1:]+(cluster_center+space_btwn_bars)
            x_acp_wil = X_axis[1:]+(cluster_center+3*space_btwn_bars)

            plt.bar(x_pp_nil, stats['point_False'][f'{stat}_mn'], bar_width, yerr=stats['point_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[3], label='point, no IL')
            plt.bar(x_cp_nil, stats['cp_False'][f'{stat}_mn'], bar_width, yerr=stats['cp_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[2], label='conformal, no IL')
            plt.bar(x_rcp_nil, stats['rcp_False'][f'{stat}_mn'], bar_width, yerr=stats['rcp_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[1], label='robust conformal, no IL')
            plt.bar(x_acp_nil, stats['acp_False'][f'{stat}_mn'], bar_width, yerr=stats['acp_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[0], label='adaptive conformal, no IL')
            plt.bar(x_pp_wil, stats['point_True'][f'{stat}_mn'], bar_width, yerr=stats['point_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[7], label='point, with IL')
            plt.bar(x_cp_wil, stats['cp_True'][f'{stat}_mn'], bar_width, yerr=stats['cp_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[6], label='conformal, with IL')
            plt.bar(x_rcp_wil, stats['rcp_True'][f'{stat}_mn'], bar_width, yerr=stats['rcp_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[5], label='robust conformal, with IL')
            plt.bar(x_acp_wil, stats['acp_True'][f'{stat}_mn'], bar_width, yerr=stats['acp_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[4], label='adaptive conformal, with IL')
            plt.xticks(X_axis, SCENARIOS)
            plt.ylabel(stat_name)
            if stat == 'steps':
                plt.ylim([0,5.5])
            elif stat in ['prec', 'rec']:
                plt.ylim([0,1.2])
            plt.savefig(os.path.join(args.save_dir, f'{args.case}_{stat}.png'))

            if stat == 'far':
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f'legend_crop.png'))

            plt.close()

            
        # results without rcp
        # X_axis = np.arange(len(SCENARIOS))*2.3
        # bar_width = 0.2
        # cluster_center = 0.45
        # space_btwn_bars = 0.3
        # colors = categorical_cmap(2,3,cmap="tab10").colors
        # ebar_size = 2
        # ebar_color = 'black'

        # for stat in ['far', 'mar', 'steps', 'prec', 'rec']:
        #     if args.case == 'safe' and stat in ['mar', 'steps']:
        #         continue

        #     stat_name = {
        #         'far':'False Alarm Rate (%)',
        #         'mar':'Missed Alarm Rate (%)',
        #         'steps':'Steps',
        #         'prec':'Precision',
        #         'rec':'Recall'
        #     }[stat]

        #     x_pp_nil = [X_axis[i]-(cluster_center+space_btwn_bars)  if i > 0 else X_axis[i]-0.30 for i in range(len(X_axis))]
        #     x_cp_nil = [X_axis[i]-cluster_center                    if i > 0 else X_axis[i] for i in range(len(X_axis))]
        #     x_acp_nil = [X_axis[i]-(cluster_center-space_btwn_bars) if i > 0 else X_axis[i]+0.30 for i in range(len(X_axis))]
        #     x_pp_wil = X_axis[1:]+(cluster_center-space_btwn_bars)
        #     x_cp_wil = X_axis[1:]+cluster_center
        #     x_acp_wil = X_axis[1:]+(cluster_center+space_btwn_bars)

        #     plt.bar(x_pp_nil, stats['point_False'][f'{stat}_mn'], bar_width, yerr=stats['point_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[2], label='point, no IL')
        #     plt.bar(x_cp_nil, stats['cp_False'][f'{stat}_mn'], bar_width, yerr=stats['cp_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[1], label='conformal, no IL')
        #     plt.bar(x_acp_nil, stats['acp_False'][f'{stat}_mn'], bar_width, yerr=stats['acp_False'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[0], label='adaptive conformal, no IL')
        #     plt.bar(x_pp_wil, stats['point_True'][f'{stat}_mn'], bar_width, yerr=stats['point_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[5], label='point, with IL')
        #     plt.bar(x_cp_wil, stats['cp_True'][f'{stat}_mn'], bar_width, yerr=stats['cp_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[4], label='conformal, with IL')
        #     plt.bar(x_acp_wil, stats['acp_True'][f'{stat}_mn'], bar_width, yerr=stats['acp_True'][f'{stat}_std'], capsize=ebar_size, ecolor=ebar_color, color=colors[3], label='adaptive conformal, with IL')
        #     plt.xticks(X_axis, SCENARIOS)
        #     plt.ylabel(stat_name)
        #     if stat == 'steps':
        #         plt.ylim([0,5.5])
        #     elif stat in ['prec', 'rec']:
        #         plt.ylim([0,1.15])
        #     plt.savefig(os.path.join(args.save_dir, f'{args.case}_{stat}.png'))

        #     if stat == 'far':
        #         plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        #         plt.tight_layout()
        #         plt.savefig(os.path.join(args.save_dir, f'legend_crop.png'))

        #     plt.close()

    else:
        print('Cannot perform statistical significance tests and plotting until entire experiment is completed.')