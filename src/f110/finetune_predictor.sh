#!/bin/bash

for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u train_predictor.py --finetune $OOD --dataset_seed $SEED > "logs/ft_predictor/ft_predictor_${OOD}_${SEED}.txt"
        python3.7 -u eval_predictor.py --finetune $OOD --dataset_seed $SEED > "logs/ft_predictor/eval_ft_predictor_${OOD}_${SEED}.txt"
    done
done