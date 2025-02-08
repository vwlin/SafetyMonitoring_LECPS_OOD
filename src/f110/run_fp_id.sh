#!/bin/bash

# point
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base models --pred_method point > "logs/fp/fp_id_point_${SEED}.txt"
    python3.7 -u select_ncs_threshold.py --quantile 0.8 --dataset_seed $SEED --f0_predictions_base models > "logs/select_ncs_threshold/select_ncs_threshold_${SEED}.txt"
done

echo "Summarizing ID - no IL, with point prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --predictions_base models --pred_method point
echo "Done"

# conformal
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base models --pred_method cp > "logs/fp/fp_id_cp_${SEED}.txt"
done

echo "Summarizing ID - no IL, with conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --predictions_base models --pred_method cp
echo "Done"

# robust conformal
EPS=0.08
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base models --pred_method rcp --epsilon $EPS > "logs/fp/fp_id_rcp_${EPS}_${SEED}.txt"
done

echo "Summarizing ID - no IL, with robust conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --predictions_base models --pred_method rcp --epsilon $EPS
echo "Done"

# adaptive conformal
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3.7 -u failure_prediction.py --scenario id --split test --dataset_seed $SEED --predictions_base models --pred_method acp > "logs/fp/fp_id_acp_${SEED}.txt"
done

echo "Summarizing ID - no IL, with adaptive conformal prediction"
python3.7 -u summarize_fp_results.py --scenario id --split test --predictions_base models --pred_method acp
echo "Done"