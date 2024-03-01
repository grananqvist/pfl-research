#!/bin/bash

# Run from the benchmarks dir

export PYTHONPATH="."

python -m asr.pytorch.train \
    --data_path ~/.cache/mlx.data/librispeech \
    --dataset common-voice-en-v13 \
    --model_name asr_ctc_transformer \
    --local_batch_size 384000 \
    --evaluation_frequency 1 \
    --cohort_size 20 \
    --training_split train \
    --validation_split dev \
    --evaluation_splits dev test other \
    --val_cohort_size 10 \
    --central_eval_batch_size 384000 \
    --local_learning_rate 0.01 \
    --central_optimizer adam \
    --learning_rate 0.1 \
    --max_sample_audio_length 384000 \
    --num_threads_data_processing 8 \
    --central_num_iterations 5 \
    --local_num_epochs 10 \
    --dummy_model_size 100
