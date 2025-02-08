#!/bin/bash

# point
for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method point > "logs/fp/fp_${OOD}_kmeans-il_point_${SEED}.txt"
    done

    echo "Summarizing ${OOD} - with IL, with point prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method point
    echo "Done"
done

# conformal
for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method cp > "logs/fp/fp_${OOD}_kmeans-il_cp_${SEED}.txt"
    done
    
    echo "Summarizing ${OOD} - with IL, with conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method cp
    echo "Done"
done

# robust conformal
EPS=0.08
for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method rcp --epsilon $EPS > "logs/fp/fp_${OOD}_kmeans-il_rcp_${EPS}_${SEED}.txt"
    done

    echo "Summarizing ${OOD} - with IL, with robust conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method rcp --epsilon $EPS
    echo "Done"
done

# adaptive conformal
for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method acp > "logs/fp/fp_${OOD}_kmeans-il_acp_${SEED}.txt"
    done
    
    echo "Summarizing ${OOD} - with IL, with adaptive conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --predictions_base "models_${OOD}" --use_incremental --cluster_method kmeans --pred_method acp
    echo "Done"
done