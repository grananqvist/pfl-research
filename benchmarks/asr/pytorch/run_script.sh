#!/bin/bash

# Run from the benchmarks dir

export PYTHONPATH="."

python -m asr.pytorch.train \
    --data_path ~/.cache/mlx.data/librispeech \
    --dataset librispeech \
    --model_name asr_ctc_transformer \
    --local_batch_size 384000 \
    --evaluation_frequency 10 \
    --cohort_size 64 \
    --training_split train-clean-100 \
    --validation_split dev-clean \
    --evaluation_splits dev-clean dev-other test-clean test-other \
    --val_cohort_size 10 \
    --central_eval_batch_size 384000 \
    --local_learning_rate 0.01 \
    --central_optimizer lamb \
    --learning_rate 0.1 \
    --max_sample_audio_length 384000 \
    --num_threads_data_processing 24 \
    --central_num_iterations 10 \
    --local_num_epochs 10 \
    --dummy_model_size 255
