#!/bin/bash

wget https://bootstrap.pypa.io/get-pip.py
python3.10 ./get-pip.py
python3.10 -m pip install poetry
poetry env use $(which python3.10)
poetry install -E pytorch -E trees
. $(dirname $(poetry run which python3.10))/activate
pip install pandas mlx-data horovod unidecode einops
# for Lamb
pip install torch_optimizer
# for LARS
pip install torchlars
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

