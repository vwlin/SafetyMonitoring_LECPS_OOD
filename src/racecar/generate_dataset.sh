#!/bin/bash

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3.7 -u generate_dataset.py --seed $SEED > "logs/generate_dataset/generate_dataset_${SEED}.txt" 
done