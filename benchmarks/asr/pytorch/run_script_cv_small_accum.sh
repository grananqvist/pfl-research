#!/bin/bash

# Run from the benchmarks dir

export PYTHONPATH="."

time python -m asr.pytorch.train \
    --data_path ~/.cache/mlx.data/cv-v13 \
    --dataset asr-hdf5 \
    --model_name asr_ctc_transformer \
    --local_batch_size 192000 \
    --grad_accumulation_steps 2 \
    --evaluation_frequency 1 \
    --cohort_size 128 \
    --training_split en-dev \
    --validation_split en-test \
    --evaluation_splits en-test \
    --val_cohort_size 0 \
    --central_eval_batch_size 1920000 \
    --local_learning_rate 0.3 \
    --central_optimizer lamb \
    --learning_rate 0.006 \
    --max_sample_audio_length 1920000 \
    --num_threads_data_processing 24 \
    --central_num_iterations 5 \
    --local_num_steps 5 \
    --local_num_epochs 5 \
    --dummy_model_size 100 \
    --central_lr_warmup_iterations 3 \
    --central_lr_decay_schedule exponential-decay \
    --central_lr_decay_gamma 0.9 \
    --local_max_grad_norm 1.0 \
    --local_privacy_mechanism norm_clipping_only \
    --local_order 2 \
    --local_privacy_clipping_policy per_layer_uniform \
    --local_privacy_clipping_bound 0.001

