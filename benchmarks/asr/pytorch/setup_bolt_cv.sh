#!/bin/bash

wget https://bootstrap.pypa.io/get-pip.py
python3.10 ./get-pip.py
python3.10 -m pip install poetry
poetry env use $(which python3.10)
poetry install -E pytorch -E trees
. $(dirname $(poetry run which python3.10))/activate
pip install pandas mlx-data horovod unidecode einops h5py wandb
# for Lamb
pip install torch_optimizer
# for LARS
pip install torchlars
pip install -i https://pypi.apple.com/simple/ apple-certifi
pip install -i https://pypi.apple.com/simple iris-ml-run turibolt

export NO_PROXY="${NO_PROXY},mlr-wandb.corp.apple.com"
export no_proxy="${no_proxy},mlr-wandb.corp.apple.com"
export WANDB_API_KEY="local-cb3b7266d097ac218cdcdbff0c8657f0c92dbcff"
export WANDB_HOST="https://mlr-wandb.corp.apple.com"
export WANDB_BASE_URL="https://mlr-wandb.corp.apple.com"
export WANDB_GROUP=pfl
#wandb login --relogin

cd benchmarks

#python dataset/asr/download_librispeech.py
#pip install awscli
shopt -s expand_aliases
alias blobby='aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300'
mkdir -p ~/.cache/mlx.data/cv-v13
#blobby s3 cp s3://pfl-research-asr-data/cv-corpus-13.0-2023-03-09-flac-en.tar ~/.cache/mlx.data/cv-v13/
blobby s3 cp s3://pfl-research-asr-data/federated-en-train.hdf5 ~/.cache/mlx.data/cv-v13/
blobby s3 cp s3://pfl-research-asr-data/federated-en-dev.hdf5 ~/.cache/mlx.data/cv-v13/
blobby s3 cp s3://pfl-research-asr-data/federated-en-test.hdf5 ~/.cache/mlx.data/cv-v13/
ls -al ~/.cache/mlx.data/cv-v13/

pip list

echo "env script:"
echo $(dirname $(poetry run which python3.10))/activate

