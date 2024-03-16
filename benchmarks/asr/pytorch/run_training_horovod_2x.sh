#!/bin/bash

NUM_PROCESSES=8
NUM_GPUS=8

# poetry shell
nvidia-smi -c 0
source $(dirname $(poetry run which python3.10))/activate
cd benchmarks
#echo "ARGUMENTS: $@"
#quoted_args="$(printf "${1+ %q}" "$@")"
horovodrun --gloo -np "$NUM_PROCESSES" -H "localhost:$NUM_GPUS" python -m asr.pytorch.train "$@"
