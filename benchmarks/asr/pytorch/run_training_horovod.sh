#!/bin/bash

# poetry shell
source $(dirname $(poetry run which python3.10))/activate
cd benchmarks
#echo "ARGUMENTS: $@"
#quoted_args="$(printf "${1+ %q}" "$@")"
horovodrun --gloo -np 4 -H localhost:4 python -m asr.pytorch.train "$@"
