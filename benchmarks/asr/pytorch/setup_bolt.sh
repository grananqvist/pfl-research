#!/bin/bash

wget https://bootstrap.pypa.io/get-pip.py
python3.10 ./get-pip.py
python3.10 -m pip install poetry
poetry env use $(which python3.10)
poetry install -E pytorch -E trees
. $(dirname $(poetry run which python3.10))/activate
pip install pandas mlx-data horovod
# for Lamb
pip install torch_optimizer
# for LARS
pip install torchlars
cd benchmarks

python dataset/asr/download_librispeech.py

pip list

echo "env script:"
echo $(dirname $(poetry run which python3.10))/activate

