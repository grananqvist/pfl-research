#!/bin/bash

NUM_PROCESSES=4
NUM_GPUS=4

source $(dirname $(poetry run which python3.10))/activate
cd benchmarks
horovodrun --gloo -np "$NUM_PROCESSES" -H "localhost:$NUM_GPUS" python -m asr.pytorch.train "$@"
