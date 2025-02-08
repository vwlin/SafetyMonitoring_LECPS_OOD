#!/bin/bash

# uncomment below lines if using raw data instead of preprocessed data

echo "Creating log directories.."
mkdir logs
mkdir logs/generate_dataset
mkdir logs/train_predictor
mkdir logs/fp
mkdir logs/select_ncs_threshold
mkdir logs/ft_predictor
mkdir logs/memories
mkdir calibration_sets

echo "Generating dataset..."
bash generate_dataset.sh

echo "Training predictor..."
bash train_predictor.sh
echo "Creating calibration sets for baselines..."
bash create_cal_set_for_baselines.sh
echo "Running failure prediction algorithm without incremental learning..."
bash run_fp_id.sh
bash run_fp_ood.sh

echo "Finetuning predictor..."
bash finetune_predictor.sh
echo "Creating memories..."
bash create_memories.sh
echo "Running failure prediction algorithm with incremental learning..."
bash run_fp_ood_il.sh

echo "Calculating final stats..."
python3.7 -u calc_final_stats.py --case all --epsilon 0.08 > logs/final_stats.txt