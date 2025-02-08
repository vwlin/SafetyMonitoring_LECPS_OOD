#!/bin/bash

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3.7 -u train_predictor.py --dataset_seed $SEED > "logs/train_predictor/train_predictor_${SEED}.txt"
    python3.7 -u eval_predictor.py --dataset_seed $SEED > "logs/train_predictor/eval_predictor_${SEED}.txt"
done