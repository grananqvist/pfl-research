#!/bin/bash

NUM_PROCESSES=8
NUM_GPUS=8

export NO_PROXY="${NO_PROXY},mlr-wandb.corp.apple.com"
export no_proxy="${no_proxy},mlr-wandb.corp.apple.com"
export WANDB_API_KEY="local-cb3b7266d097ac218cdcdbff0c8657f0c92dbcff"

# poetry shell
nvidia-smi -c 0
source $(dirname $(poetry run which python3.10))/activate
cd benchmarks
#echo "ARGUMENTS: $@"
#quoted_args="$(printf "${1+ %q}" "$@")"
horovodrun --gloo -np "$NUM_PROCESSES" -H "localhost:$NUM_GPUS" python -m asr.pytorch.train "$@"
