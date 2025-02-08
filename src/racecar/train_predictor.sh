#!/bin/bash

cd AgentFormer

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    # train f0 and evaluate on validation ID data
    python3.7 -u train.py --cfg racetrack_agentformer_pre --gpu $1 --dataset_seed $SEED > "../logs/train_predictor/train_predictor_${SEED}.txt"
    python3.7 -u test.py --cfg racetrack_agentformer_pre --data_eval val --epochs 2,4,6,8,10,12,14,16,18,20 --gpu $1 --dataset_seed $SEED > "../logs/train_predictor/eval_predictor_${SEED}.txt"

    # test best f0 epoch on test ID data
    python3.7 -u test.py --cfg racetrack_agentformer_pre --data_eval test --metrics_file "results_s${SEED}/racetrack_agentformer_pre/validation_metrics.pkl" --gpu $1 --dataset_seed $SEED > "../logs/train_predictor/test_predictor_${SEED}.txt"

    # prepare results folders for OOD data
    python3.7 prep_ood_results_dir.py --train_cfg racetrack_agentformer_pre --train_step ogtest --dataset_seed $SEED

    # test best f0 epoch on train/test OOD data
    for OOD in 2 3 4 5
    do
        python3.7 -u test.py --cfg "racetrack_vehicles_${OOD}_ogtest" --data_eval train --metrics_file "results_s${SEED}/racetrack_agentformer_pre/validation_metrics.pkl" --gpu $1 --dataset_seed $SEED > "../logs/train_predictor/test_predictor_v${OOD}_tr_${SEED}.txt"
        python3.7 -u test.py --cfg "racetrack_vehicles_${OOD}_ogtest" --data_eval test --metrics_file "results_s${SEED}/racetrack_agentformer_pre/validation_metrics.pkl" --gpu $1 --dataset_seed $SEED > "../logs/train_predictor/test_predictor_v${OOD}_te_${SEED}.txt"
    done
done

cd ..