# Safety Monitoring for Learning-Enabled Cyber-Physical Systems in Out-of-Distribution Scenarios
This is the repository for the paper Safety Monitoring for Learning-Enabled Cyber-Physical Systems in Out-of-Distribution Scenarios, accepted for presentation at ICCPS 2025.

*The safety of learning-enabled cyber-physical systems is hindered by the well-known vulnerabilities of deep neural networks to out-ofdistribution (OOD) inputs. Existing literature has sought to monitor the safety of such systems by detecting OOD data. However, such approaches have limited utility, as the presence of an OOD input does not necessarily imply the violation of a desired safety property. We instead propose to directly monitor safety in a manner that is itself robust to OOD data. To this end, we predict violations of signal temporal logic safety specifications based on predicted future trajectories. Our safety monitor additionally uses a novel combination of adaptive conformal prediction and incremental learning. The former obtains probabilistic prediction guarantees even on OOD data, and the latter prevents overly conservative predictions. We evaluate the efficacy of the proposed approach in two case studies on safety monitoring: 1) predicting collisions of an F1Tenth car with static obstacles, and 2) predicting collisions of a race car with multiple dynamic obstacles. We find that adaptive conformal prediction obtains theoretical guarantees where other uncertainty quantification methods fail to do so. Additionally, combining adaptive conformal prediction and incremental learning for safety monitoring achieves high recall and timeliness while reducing loss in precision. We achieve these results even in OOD settings and outperform alternative methods.*

## Requirements
The published results were obtained on a machine running Debian 11.0 with 96 cores. With these specifications, the expected total runtime for all 10 seeded trials is about 10 hours for each case study.

## Environment and Dependencies

1. Install [Docker](https://docs.docker.com/get-started/get-docker/) on your machine.
2. A Dockerfile is provided. Either pull the *safety_monitoring* (v1.0) [Docker image](https://hub.docker.com/r/vwlin/safety_monitoring) from Docker Hub (recommended):
    ```
    docker pull vwlin/safety_monitoring:v1.0
    ```
    or build the image locally:
    ```
    docker build -t safety_monitoring .
    ```
    The Docker image is about 40 GB.
4. Run a Docker container based on this image:
    ```
    docker run -it --name my_container vwlin/safety_monitoring:v1.0 /bin/bash
    ```
    The experiments can now be reproduced by running scripts within this container. Use the command `exit` to exit the container. To reenter the container, run
    ```
    docker start my_container
    ```
    ```
    docker exec -it my_container /bin/bash
    ```
    To easily view results figures, copy them from the docker container. For example, to export the cartpole results figures, run
    ```
    docker cp my_container:/home/safety_monitoring/src/cartpole/figs PATH/TO/SafetyMonitoring_LECPS_OOD/src/cartpole
    ```

## Simulation Data and Pretrained Models
Pre-processed simulation data and pretrained trajectory predictor models are provided.

1. Download the [simulation data](https://figshare.com/articles/dataset/Data_from_Safety_Monitoring_for_Learning-Enabled_Cyber-Physical_Systems_in_Out-of-Distribution_Scenarios/28355438) and place each individual .tar.gz file in the top level directory of this repository (i.e., PATH/TO/SafetyMonitoring_LECPS_OOD/*.tar.gz).
2. Download the [models](https://figshare.com/articles/software/Models_from_Safety_Monitoring_for_Learning-Enabled_Cyber-Physical_Systems_in_Out-of-Distribution_Scenarios/28366880) and place each individual .tar.gz file in the top level directory of this repository (i.e., PATH/TO/SafetyMonitoring_LECPS_OOD/*.tar.gz). The compressed files together are about 6GB.
3. Run the setup script. This script will unpack the data/models and copy them to the Docker container. Be sure to replace `PATH/TO/SafetyMonitoring_LECPS_OOD` with your local path. This process may take several minutes, due to the size of the files.
    ```
    ./setup_data_models.sh PATH/TO/SafetyMonitoring_LECPS_OOD my_container
    ```

For more information on generating simulations for the F1/10 case study, see the [autonomous car verification](https://github.com/rivapp/autonomous_car_verification) repository by Radoslav Ivanov. Simulations were gathered using the [pre-trained DDPG_L21_64x64_C1 controller](https://github.com/rivapp/autonomous_car_verification/blob/78f7bee5aca2e7f7008e9dccd9df0088a079b398/dnns/DDPG_L21_64x64_C1.yml).

For more information on generating simulations for the race car case study, see [here](https://github.com/vwlin/OOD_Racetrack_Simulation). Note that the race car simulation code requires a different set of dependencies from this repository due to conflicts between the AgentFormer and HighwayEnv requirements.

## Reproducing our Motivating Cartpole Example
Run the following commands to reproduce our cartpole example. Results figures will be dumped to a `figs` directory.

1. `cd src/cartpole`
2. `python3.7 cartpole_ood.py --ood_type TYPE`

To view options for the `ood_type` tag, run the script with the `--help` tag. For the cartpole example, we use a [pre-trained Deep Q-Network from Mohit Pilkhan](https://github.com/mahakal001/reinforcement-learning/tree/master/cartpole-dqn) that is provided in the directory.

## Reproducing our F1/10 and Race Car Case Studies
In either the `src/f110` or `src/racecar` directories, run the following commands in sequence to reproduce our results.

1. Train and evaluate our safety monitor, reproducing our main results (Tables 1-3, Figure 6). Results will be dumped to a `src/*/logs/final_stats.txt` file and a `src/*/figs` directory.
    ```
    ./run_pipeline.sh
    ```
2. Reproduce our empirical evaluations of ACP, CP, and RCP (Figure 5, Table 6). Figure 5 will be dumped to a `src/*/failure_prediction_results/acp_figures` directory.
    ```
    python3.7 test_acp_all_scenarios.py
    python3.7 test_cp_all_scenarios.py
    python3.7 test_rcp_all_scenarios.py --epsilon EPSILON
    ```
    Run the scripts with the `--use_incremental` tag to evaluate with incremental learning. Refer to Section 7 of our paper for the appropriate choice of epsilon.
3. Reproduce our evaluations of the trajectory predictors (Table 4).
    ```
    python3.7 get_predictor_stats.py --finetune SCENARIO
    ```
    To view options for the `finetune` tag, run the script with the `--help` tag.
4. Reproduce our empirical evaluations of the non-conformity score distributions (Table 4, Table 5, Figure 8). Figure 8 will be dumped to a `src/*/failure_prediction_results/rcp_figures`
    ```
    python3.7 estimate_rcp_eps.py
    ```

## Additional Notes
The above scripts reproduce our results with pretrained models. To train new models and run our safety monitor with these, it is highly recommended to use GPUs. We trained our models on a machine running CUDA 12.3 and 24 GB of GPU memory. With these specifications, the expected runtime for a single seeded trial of the F1/10 case study is about 2.5 hours. The expected runtime for a single seeded trial of the race car case study is several days.
1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your machine.
2. Download the [raw simulation data](https://figshare.com/articles/dataset/Data_from_Safety_Monitoring_for_Learning-Enabled_Cyber-Physical_Systems_in_Out-of-Distribution_Scenarios/28355438) and unpack it to the raw_data directory.
3. Build a Docker image with GPU capability.
    ```
    docker build -t safety_monitoring_gpu -f Dockerfile.gpu .
    ```
4. Run a Docker container based on this image:
    ```
    docker run -it --gpus all --name my_container_gpu safety_monitoring_gpu /bin/bash
    ```
5. Run the following script in the Docker container.
    ```
    ./run_pipeline_scratch.sh
    ```
**Important Note: These evaluations are highly dependent on random seeding. Due to differences in seeding across machines, these evaluations will reproduce our results and conclusions, but may not exactly match our published values.**