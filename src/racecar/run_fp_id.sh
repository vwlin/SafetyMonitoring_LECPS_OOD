#!/bin/bash

# point
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_agentformer_pre" --pred_method point --verbose > "logs/fp/fp_id_point_${SEED}.txt"

    python3.7 -u select_ncs_threshold.py --quantile 0.5 --dataset_seed $SEED --metrics_file "AgentFormer/results_s${SEED}/racetrack_agentformer_pre/validation_metrics.pkl" > "logs/select_ncs_threshold/selection_stats_${SEED}.txt"
done

echo "Summarizing ID - no IL, with point prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --pred_method point
echo "Done"

# conformal
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_agentformer_pre" --pred_method cp --verbose > "logs/fp/fp_id_cp_${SEED}.txt"
done

echo "Summarizing ID - no IL, with conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --pred_method cp
echo "Done"

# robust conformal
EPS=0.03
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_agentformer_pre" --pred_method rcp --epsilon $EPS --verbose > "logs/fp/fp_id_rcp_${EPS}_${SEED}.txt"
done

echo "Summarizing ID - no IL, with robust conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --pred_method rcp --epsilon $EPS
echo "Done"

# adaptive conformal
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_agentformer_pre" --pred_method acp --verbose > "logs/fp/fp_id_acp_${SEED}.txt"
done

echo "Summarizing ID - no IL, with adaptive conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --pred_method acp
echo "Done"
