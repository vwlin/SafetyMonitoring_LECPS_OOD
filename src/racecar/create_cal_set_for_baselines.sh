#!/bin/bash

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u create_cal_set_for_baselines.py --dataset_seed $SEED
done