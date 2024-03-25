#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_PROCESSES=$NUM_GPUS

echo "NUM_GPUS=${NUM_GPUS}"

source $(dirname $(poetry run which python3.10))/activate

export NO_PROXY="${NO_PROXY},mlr-wandb.corp.apple.com"
export no_proxy="${no_proxy},mlr-wandb.corp.apple.com"
export WANDB_API_KEY="local-cb3b7266d097ac218cdcdbff0c8657f0c92dbcff"
export WANDB_HOST="https://mlr-wandb.corp.apple.com"
export WANDB_BASE_URL="https://mlr-wandb.corp.apple.com"
export WANDB_GROUP=pfl

cd benchmarks
horovodrun --gloo -np "$NUM_PROCESSES" -H "localhost:$NUM_GPUS" python -m asr.pytorch.train "$@"
