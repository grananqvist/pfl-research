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
        help='Tar dataset file with the training data. '
             'The path is relative to data_path.')
    argument_parser.add_argument(
        '--evaluation_splits',
        nargs='+',
        default=[
            "dev-clean", "dev-other", "test-clean", "test-other"],
        help='List of data splits to use for evaluation on the server. '
             'E.g., "test-clean test-other".')

    # Model arguments
    argument_parser.add_argument(
        '--cape',
        action=store_bool,
        default=True,
        help='If enabled, use cape embedding.')
    argument_parser.add_argument(
        "--layer_drop",
        type=float,
        default=0.3,
        help='Model layer drop.')  # TODO: Describe better
    argument_parser.add_argument(
        "--dropout",
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

    # TODO: Add LR decay and warmup arguments (sever, optionally local)

    return argument_parser
