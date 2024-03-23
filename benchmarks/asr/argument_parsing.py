# Copyright © 2023-2024 Apple Inc.
import argparse
import os

from utils.argument_parsing import add_dnn_training_arguments, add_mechanism_arguments, store_bool


def add_asr_arguments(argument_parser):
    argument_parser = add_dnn_training_arguments(argument_parser)
    argument_parser = add_mechanism_arguments(argument_parser)

    # Optimizer arguments
    argument_parser.add_argument('--central_optimizer',
                                 choices=['sgd', 'adam', 'lamb', 'lars'],
                                 default='sgd',
                                 help='Optimizer for central updates')

    known_args, _ = argument_parser.parse_known_args()
    if known_args.central_optimizer in ['adam', 'lamb']:
        argument_parser.add_argument(
            '--adaptivity_degree',
            type=float,
            default=0.01,
            help='Degree of adaptivity (eps) in adaptive server optimizer.')
    argument_parser.add_argument(
        '--local_max_grad_norm',
        type=float,
        default=None,
        help='Gradient clipping bound in local SGD training.')

    # Dataset arguments
    argument_parser.add_argument(
        '--training_split',
        type=str,
        default="train-clean-100",
        help='Data split to use for training the model using PFL.')
    argument_parser.add_argument(
        '--validation_split',
        type=str,
        default="dev-clean",
        help=
        'Data split to use for validating the model in a federated setting.')
    argument_parser.add_argument(
        '--evaluation_splits',
        nargs='+',
        default=["dev-clean", "dev-other", "test-clean", "test-other"],
        help='List of data splits to use for central evaluation on the server. '
        'E.g., "test-clean test-other".')
    argument_parser.add_argument("--max_sample_audio_length",
                                 type=int,
                                 default=None,
                                 help='Maximum length of audio for a sample '
                                 'to be used in training or evaluation. '
                                 'All samples with longer audio will be '
                                 'filtered out.')
    argument_parser.add_argument(
        "--num_threads_data_processing",
        type=int,
        default=4,
        help='Number of threads for processing the data.')
    argument_parser.add_argument("--dummy_model_size",
                                 type=int,
                                 default=1,
                                 help='Model size in millions (must be >= 1).')
    argument_parser.add_argument("--lazy_load_audio",
                                 action=store_bool,
                                 default=False,
                                 help='Whether to load audio only once the user dataset or '
                                      'central dataset is created. This typically saves '
                                      'memory for large federated datasets but possibly '
                                      'prolongs the cohort processing.')

    # Model arguments
    argument_parser.add_argument('--cape',
                                 action=store_bool,
                                 default=True,
                                 help='If enabled, use cape embedding.')
    argument_parser.add_argument(
        "--layer_drop", type=float, default=0.3,
        help='Model layer drop.')  # TODO: Describe better
    argument_parser.add_argument("--dropout",
                                 type=float,
                                 default=0.3,
                                 help='Dropout.')  # TODO: Describe better
    argument_parser.add_argument(
        "--additional_chars",
        default=None,
        type=str,
        help='Use these additional characters to construct the trie used for '
        'tokenization. Models with different characters are currently '
        'incompatible. E.g. "-ü".')

    # Training and evaluation arguments

    argument_parser.add_argument(
        "--batch_strategy",
        default='dynamic',
        type=str,
        choices=['dynamic', 'static'],
        help='Batch strategy to use for local training.')

    argument_parser.add_argument(
        '--amp_dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Float format in mixed precision training.')

    argument_parser.add_argument(
        '--grad_scaling',
        action=store_bool,
        default=False,
        help='Whether enable gradient scaling when using'
        ' mixed precision training.')

    argument_parser.add_argument(
        '--model_dtype_same_as_amp',
        action=store_bool,
        default=False,
        help='Cast the model weights precision to the same as used in '
        'autocast. This saves memory but may cause divergence due to '
        'lower precision.')

    argument_parser.add_argument(
        '--grad_accumulation_steps',
        type=int,
        default=1,
        help='Effective local batch size is local batch size '
        'multiplied by this number.')

    argument_parser.add_argument(
        '--use_torch_compile',
        action=store_bool,
        default=False,
        help='Whether to use `torch.compile` on the PyTorch module.')

    # TODO: Add LR decay and warmup arguments (sever, optionally local)
    argument_parser.add_argument(
        '--central_lr_warmup_iterations',
        type=int,
        default=0,
        help='Number of iterations to warmup central learning rate.')
    argument_parser.add_argument('--central_lr_schedule',
                                 choices=['constant', 'step-decay', 'exponential-decay'],
                                 default='constant',
                                 help='Central learning rate schedule. Warmup is inserted at the '
                                      'beginning of the chosen schedule, if warmup is used as '
                                      'specified by the flag --central_lr_warmup_iterations.')

    known_args, _ = argument_parser.parse_known_args()
    if known_args.central_lr_schedule == 'step-decay':
        argument_parser.add_argument('--central_lr_step_decay_iterations',
                                     type=int,
                                     default=500,
                                     help='Number of central iterations for central learning '
                                          'rate step decay.')
    if known_args.central_lr_schedule in ['step-decay', 'exponential-decay']:
        argument_parser.add_argument('--central_lr_decay_gamma',
                                     type=float,
                                     default=0.5,
                                     help='Learning rate decay factor for step or exponential '
                                          'central learning rate decay.')

    return argument_parser


def get_central_lr_schedular(arguments, central_optimizer):
    import torch.optim.lr_scheduler
    if arguments.central_lr_warmup_iterations > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=central_optimizer,
            start_factor=1.0 / (arguments.central_lr_warmup_iterations + 1),
            end_factor=1.0,
            total_iters=arguments.central_lr_warmup_iterations,
        )
    else:
        warmup_scheduler = None

    if arguments.central_lr_schedule == 'step-decay':
        central_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=central_optimizer,
            step_size=arguments.central_lr_step_decay_iterations,
            gamma=arguments.central_lr_decay_gamma,
        )
    elif arguments.central_lr_schedule == 'exponential-decay':
        central_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=central_optimizer,
            gamma=arguments.central_lr_decay_gamma,
        )
    elif arguments.central_lr_schedule == 'constant':
        central_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer=central_optimizer,
            factor=1.0,
            total_iters=arguments.central_num_iterations - arguments.central_lr_warmup_iterations,
        )
    else:
        raise ValueError(f'Central learning rate schedule '
                         f'{arguments.central_lr_schedule} not implemented')

    if warmup_scheduler:
        central_lr_scheduler_full = torch.optim.lr_scheduler.SequentialLR(
            central_optimizer,
            schedulers=[warmup_scheduler, central_lr_scheduler],
            milestones=[arguments.central_lr_warmup_iterations])
    else:
        central_lr_scheduler_full = central_lr_scheduler

    return central_lr_scheduler_full
