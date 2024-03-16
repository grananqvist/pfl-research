#!/bin/bash

# Run from the benchmarks dir

export PYTHONPATH="."

python -m asr.pytorch.train \
    --data_path ~/.cache/mlx.data/cv-v13 \
    --dataset common-voice-en-v13 \
    --model_name asr_ctc_transformer \
    --local_batch_size 384000 \
    --evaluation_frequency 1 \
    --cohort_size 20 \
    --training_split en-dev \
    --validation_split en-test \
    --evaluation_splits en-dev en-test \
    --val_cohort_size 10 \
    --central_eval_batch_size 384000 \
    --local_learning_rate 0.01 \
    --central_optimizer adam \
    --learning_rate 0.1 \
    --max_sample_audio_length 384000 \
    --num_threads_data_processing 24 \
    --central_num_iterations 5 \
    --local_num_epochs 10 \
    --dummy_model_size 100
