import pickle
import numpy as np
from scipy.stats import gaussian_kde
import argparse
import matplotlib.pyplot as plt

from utils.utils import *

# augmented from https://github.com/SAIDS-Lab/Robust-Conformal-Prediction-for-STL-Runtime-Verification-under-Distribution-Shift/blob/main/Franka_Manipulator/demo.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fp_results_root', default='failure_prediction_results', type=str, help='Directory for results of failure prediction.')
    parser.add_argument('--predictions_base', type=str, default='AgentFormer', help='Base path to model predictions.')
    args = parser.parse_args()

    print(args)
    np.random.seed(0)

    save_dir = f'{args.fp_results_root}/rcp_figures'
    os.makedirs(save_dir, exist_ok=True)

    with open('logs/select_ncs_threshold/taus.pkl', 'rb') as f:
        saved_taus = pickle.load(f)
    tau = saved_taus[0]

    for scenario in ['id', 'vehicles_2', 'vehicles_3', 'vehicles_4', 'vehicles_5']:
        print(scenario)
        
        for il in [False, True]:
            if scenario == 'id' and il == True:
                continue

            TVDs = []
            areas = []
            for seed in range(10):
                print(seed)

                cal_pth = f'calibration_sets/calibration_set_{seed}.pkl'
                with open(cal_pth, 'rb') as f:
                    cal_ncs = pickle.load(f)

                # update predictions_base argument based on best epoch
                if not il:
                    base_pth = os.path.join(args.predictions_base, f'results_s{seed}/racetrack_agentformer_pre')
                else:
                    base_pth = os.path.join(args.predictions_base, f'results_s{seed}/racetrack_{scenario}_fttest')
                metrics_file = os.path.join(base_pth, f'validation_metrics.pkl')
                with open(metrics_file, 'rb') as f:
                    metrics = pickle.load(f)
                best_epoch = min(metrics, key=lambda x: metrics[x]['ADE'])
                predictions_base = os.path.join(base_pth, f'/results/epoch_{best_epoch:04d}')

                # get rhos
                if scenario == 'id':
                    fp_dir = make_fp_results_dir(args.fp_results_root, seed, 'id', 'test', predictions_base, 'pred',
                                                            False, 'kmeans', 'acp', None)
                else:
                    if not il:
                        fp_dir = make_fp_results_dir(args.fp_results_root, seed, scenario, 'test', predictions_base, 'pred',
                                                False, 'kmeans', 'acp', None)
                    else:
                        fp_dir = make_fp_results_dir(args.fp_results_root, seed, scenario, 'test', predictions_base, 'pred',
                                                True, 'kmeans', 'acp', None)
                                                            
                with open(os.path.join(fp_dir, 'acp_data.pkl'), 'rb') as f:
                    acp_info = pickle.load(f)
                gamma = acp_info.pop('gamma')
                delta = acp_info.pop('delta')

                ncs = []
                for e in list(acp_info.keys()):
                    far = acp_info[e].pop('far')
                    mar = acp_info[e].pop('mar')
                    fp = acp_info[e].pop('false_pos')
                    tn = acp_info[e].pop('true_neg')
                    fn = acp_info[e].pop('false_neg')
                    tp = acp_info[e].pop('true_pos')
                    steps_metric = acp_info[e].pop('steps')

                    # ncs += [(acp_info[e][t]['rho_hat']-acp_info[e][t]['rho']) for t in acp_info[e].keys()]
                    t = list(acp_info[e].keys())[np.random.randint(len(acp_info[e].keys()))]
                    ncs.append((acp_info[e][t]['rho_hat']-acp_info[e][t]['rho']))

                # estimated densities with gaussian KDE
                cal_density = gaussian_kde(cal_ncs)
                inf_density = gaussian_kde(ncs)

                # evaluate densities
                lower_bound = np.min(np.concatenate((cal_ncs, ncs)))
                upper_bound = np.max(np.concatenate((cal_ncs, ncs)))
                step_size = (upper_bound - lower_bound) / 20000
                ncs_to_evaluate = np.arange(lower_bound, upper_bound + step_size, step_size)
                cal_probs = cal_density.evaluate(ncs_to_evaluate)
                inf_probs = inf_density.evaluate(ncs_to_evaluate)

                if seed == 0:
                    if not il:
                        plt.figure(figsize=(4,4))
                        if scenario != 'id':
                            plt.axvline(-tau, linestyle='--', color='grey', label=r'$\pm \tau$')
                            plt.axvline(tau, linestyle='--', color='grey')
                        plt.plot(ncs_to_evaluate, cal_probs, label='calibration')
                    plt.plot(ncs_to_evaluate, inf_probs, label='with IL' if il else 'without IL')\

                # calc area outside of [-tau, tau]
                low_error = np.arange(-tau, tau + step_size, step_size)
                low_error_probs = inf_density(low_error)
                areas.append(1 - np.trapz(low_error_probs, low_error))

                # numerically estimate total variation distance (no IL)
                divergence = 0
                for i in range(len(ncs_to_evaluate) - 1):
                    y_front = 0.5 * abs(cal_probs[i] - inf_probs[i])
                    y_back = 0.5 * abs(cal_probs[i + 1] - inf_probs[i + 1])
                    divergence += ((y_front + y_back) * step_size / 2)
                TVDs.append(divergence)
            
            descriptor = 'with' if il else 'without'
            print(f'{scenario} {descriptor} IL, area outside [-tau,tau]:\tmean {np.mean(areas):.4f}, std {np.std(areas):.4f}')
            print(f'{scenario} {descriptor} IL, TV:\tmean {np.mean(TVDs):.4f}, std {np.std(TVDs):.4f}')

        plt.xlabel('NCS')
        plt.ylabel('p(NCS)')
        plt.legend()
        plt.title(scenario)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{scenario}.png')
        plt.close()