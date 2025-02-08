#!/bin/bash

# point
for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
do
    for SEED in 0 1 2 3 4 5 6 7 8 9
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest" --use_incremental --cluster_pth "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest/pseudo_memories.pkl" --f0_predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_ogtest" --pred_method point --verbose > "logs/fp/fp_${OOD}_kmeans-il_point_${SEED}.txt"
    done

    echo "Summarizing ${OOD} - with IL, with point prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --use_incremental --cluster_method kmeans --pred_method point
    echo "Done"
done

# conformal
for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
do
    for SEED in 0 1 2 3 4 5 6 7 8 9
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest" --use_incremental --cluster_pth "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest/pseudo_memories.pkl" --f0_predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_ogtest" --pred_method cp --verbose > "logs/fp/fp_${OOD}_kmeans-il_cp_${SEED}.txt"
    done
    
    echo "Summarizing ${OOD} - with IL, with conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --use_incremental --cluster_method kmeans --pred_method cp
    echo "Done"
done

# robust conformal
EPS=0.03
for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
do
    for SEED in 0 1 2 3 4 5 6 7 8 9
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest" --use_incremental --cluster_pth "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest/pseudo_memories.pkl" --f0_predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_ogtest" --pred_method rcp --epsilon $EPS --verbose > "logs/fp/fp_${OOD}_kmeans-il_rcp_${EPS}_${SEED}.txt"
    done

    echo "Summarizing ${OOD} - with IL, with robust conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --use_incremental --cluster_method kmeans --pred_method rcp --epsilon $EPS
    echo "Done"
done

# adaptive conformal
for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
do
    for SEED in 0 1 2 3 4 5 6 7 8 9
    do
        python3.7 -u failure_prediction.py --scenario $OOD --split test --dataset_seed $SEED --predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest" --use_incremental --cluster_pth "AgentFormer/results_s${SEED}/racetrack_${OOD}_fttest/pseudo_memories.pkl" --f0_predictions_base "AgentFormer/results_s${SEED}/racetrack_${OOD}_ogtest" --pred_method acp --verbose > "logs/fp/fp_${OOD}_kmeans-il_acp_${SEED}.txt"
    done
    
    echo "Summarizing ${OOD} - with IL, with adaptive conformal prediction"
    python3.7 -u summarize_fp_results.py --scenario $OOD --split test --use_incremental --cluster_method kmeans --pred_method acp
    echo "Done"
done
