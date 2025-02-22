#!/bin/bash

# unpack data
mkdir "${1}/src/racecar/AgentFormer/datasets"
tar -xzvf f110_processed_data.tar.gz -C "${1}/src/f110"
tar -xzvf racecar_processed_data.tar.gz -C "${1}/src/racecar/AgentFormer/datasets"

# copy data to Docker container
docker cp "${1}/src/f110/data" "${2}:/home/safety_monitoring/src/f110"
docker cp "${1}/src/racecar/AgentFormer/datasets" "${2}:/home/safety_monitoring/src/racecar/AgentFormer/datasets"

# unpack models
tar -xzvf f110_models.tar.gz -C "${1}/src/f110"
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    tar -xzvf "racecar_models_s${SEED}.tar.gz" -C "${1}/src/racecar/AgentFormer"
done

# copy models to Docker container
docker cp "${1}/src/f110/models" "${2}:/home/safety_monitoring/src/f110"
docker cp "${1}/src/f110/models_ood_0.0_3" "${2}:/home/safety_monitoring/src/f110"
docker cp "${1}/src/f110/models_ood_0.0_5" "${2}:/home/safety_monitoring/src/f110"
docker cp "${1}/src/f110/models_ood_0.9_0" "${2}:/home/safety_monitoring/src/f110"
docker cp "${1}/src/f110/models_ood_1.0_0" "${2}:/home/safety_monitoring/src/f110"
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    docker cp "${1}/src/racecar/AgentFormer/results_s${SEED}" "${2}:/home/safety_monitoring/src/racecar/AgentFormer/"
done