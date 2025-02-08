FROM python:3.7

RUN apt-get update
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -U setuptools --no-cache-dir

COPY ./ /home/safety_monitoring

WORKDIR /home/safety_monitoring

RUN python3.7 --version

RUN python3.7 -m pip install torch==1.8.1 torchvision==0.9.1 gym==0.26.2 scikit-learn==0.22.1 kneed similaritymeasures bayesian-torch
RUN python3.7 -m pip install -r src/racecar/AgentFormer/requirements.txt