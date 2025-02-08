#!/bin/bash

for OOD in 'vehicles_2' 'vehicles_3' 'vehicles_4' 'vehicles_5'
do
    for SEED in 0 1 2 3 4 5 6 7 8 9
    do
        python3.7 -u generate_pseudo_memories.py --scenario $OOD --dataset_seed $SEED > "logs/memories/kmeans_clustering_${OOD}_${SEED}.txt"
    done
done