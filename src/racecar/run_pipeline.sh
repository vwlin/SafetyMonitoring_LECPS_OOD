#!/bin/bash

echo "Creating log directories.."
mkdir logs
mkdir logs/fp
mkdir logs/select_ncs_threshold
mkdir calibration_sets

echo "Creating calibration sets for baselines..."
bash create_cal_set_for_baselines.sh

echo "Running failure prediction algorithm without incremental learning..."
bash run_fp_id.sh
bash run_fp_ood.sh

echo "Running failure prediction algorithm with incremental learning..."
bash run_fp_ood_il.sh

echo "Calculating final stats..."
python3.7 -u calc_final_stats.py --case all > logs/final_stats.txt