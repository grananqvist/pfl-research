#!/bin/bash

apt-get update
#apt-get install python3.10

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
export PATH="/root/miniforge3/bin:$PATH"
eval "$(conda shell.bash activate)"
conda install python=3.10

echo "python3.10 --version: $(python3.10 --version)"
#alias python3.10=python3
wget https://bootstrap.pypa.io/get-pip.py
python3.10 ./get-pip.py
python3.10 -m pip install poetry
poetry env use $(which python3.10)
poetry install -E pytorch -E tf -E trees
. $(dirname $(poetry run which python3.10))/activate
pip install nvtx

HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_TENSORFLOW=0 python -m pip install \
		--progress-bar off --no-cache-dir horovod[pytorch]
pip install pandas mlx-data unidecode einops h5py wandb awscli
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

