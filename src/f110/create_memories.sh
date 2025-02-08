#!/bin/bash

for OOD in 'ood_0.0_3' 'ood_0.0_5' 'ood_0.9_0' 'ood_1.0_0'
do
    for SEED in 1 2 3 4 5 6 7 8 9 10
    do
        python3.7 -u generate_pseudo_memories.py --finetune $OOD --dataset_seed $SEED > "logs/memories/kmeans_clustering_${OOD}_${SEED}.txt"
    done
done