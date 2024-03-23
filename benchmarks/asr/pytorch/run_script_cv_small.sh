#!/bin/bash

# Run from the benchmarks dir

export PYTHONPATH="."

python -m asr.pytorch.train \
    --data_path ~/.cache/mlx.data/cv-v13 \
    --dataset asr-hdf5 \
    --model_name asr_ctc_transformer \
    --local_batch_size 384000 \
    --evaluation_frequency 1 \
    --cohort_size 128 \
    --training_split en-dev \
    --validation_split en-test \
    --evaluation_splits en-dev en-test \
    --val_cohort_size 0 \
    --central_eval_batch_size 768000 \
    --local_learning_rate 0.01 \
    --central_optimizer lamb \
    --learning_rate 0.006 \
    --max_sample_audio_length 384000 \
    --num_threads_data_processing 24 \
    --central_num_iterations 5 \
    --local_num_epochs 10 \
    --dummy_model_size 100 \
    --central_lr_warmup_iterations 3 \
    --central_lr_schedule exponential-decay \
    --central_lr_decay_gamma 0.9 \
    --local_max_grad_norm 1.0

