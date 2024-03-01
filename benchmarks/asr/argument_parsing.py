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
    if known_args.central_optimizer == 'adam':
        argument_parser.add_argument(
            '--adaptivity_degree',
            type=float,
            default=0.01,
            help='Degree of adaptivity (eps) in adaptive server optimizer.')

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
    argument_parser.add_argument(
        "--dummy_model_size",
        type=int,
        default=1,
        help='Model size in millions (must be >= 1).')

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
        "--local_batch_strategy",
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

    return argument_parser
