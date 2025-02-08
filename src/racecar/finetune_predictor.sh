#!/bin/bash

cd AgentFormer

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    # prepare results folders for OOD data
    python3.7 prep_ood_results_dir.py --train_cfg racetrack_agentformer_pre --train_step ft --dataset_seed $SEED

    # train f1 and evaluate on validation high NCS OOD data
    for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
    do
        python3.7 -u train.py --cfg "racetrack_${OOD}_ft" --gpu $1 --metrics_file "results_s${SEED}/racetrack_agentformer_pre/validation_metrics.pkl" --dataset_seed $SEED > "../logs/ft_predictor/ft_predictor_${OOD}_${SEED}.txt"
        python3.7 -u test.py --cfg "racetrack_${OOD}_ft" --data_eval val --epochs 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80 --gpu $1 --dataset_seed $SEED > "../logs/ft_predictor/eval_predictor_${OOD}_${SEED}.txt"

        # prepare results folders for OOD data
        python3.7 prep_ood_results_dir.py --train_cfg racetrack_${OOD}_ft --train_step fttest --dataset_seed $SEED --ood_type $OOD

        # test best f1 epoch on test OOD data
        python3.7 -u test.py --cfg "racetrack_${OOD}_fttest" --data_eval test --metrics_file "results_s${SEED}/racetrack_${OOD}_ft/validation_metrics.pkl" --gpu $1 --dataset_seed $SEED > "../logs/ft_predictor/test_predictor_${OOD}_te_${SEED}.txt"

    done

done

cd ..