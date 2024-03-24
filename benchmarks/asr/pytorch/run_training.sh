#!/bin/bash

export NO_PROXY="${NO_PROXY},mlr-wandb.corp.apple.com"
export no_proxy="${no_proxy},mlr-wandb.corp.apple.com"
export WANDB_API_KEY="local-cb3b7266d097ac218cdcdbff0c8657f0c92dbcff"

# poetry shell
source $(dirname $(poetry run which python3.10))/activate
cd benchmarks
#echo "ARGUMENTS: $@"
#quoted_args="$(printf "${1+ %q}" "$@")"
python -m asr.pytorch.train "$@"
