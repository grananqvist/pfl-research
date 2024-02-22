import argparse
import logging

import argparse
import logging

import numpy as np
import torch
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_pytorch
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_filepath_arguments,
    add_seed_arguments,
    add_weighting_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
    parse_weighting_strategy,
)
from utils.callback.pytorch import CentralLRDecay
from utils.logging import init_logging

from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import (
    AggregateMetricsToDisk,
    CentralEvaluationCallback,
    ModelCheckpointingCallback,
    StopwatchCallback,
    TrackBestOverallMetrics,
    WandbCallback,
)
from pfl.model.pytorch import PyTorchModel
from pfl.privacy import CentrallyAppliedPrivacyMechanism

from ..argument_parsing import add_asr_arguments
from ..utils import construct_eng_char_trie_for_ctc

def main():
    init_logging(logging.DEBUG)
    maybe_inject_arguments_from_config()
    logger = logging.getLogger(name=__name__)

    # Create argument parser and parse arguments.
    logger.info('Parsing arguments')
    parser = argparse.ArgumentParser(
        description=
        'Train a model using private federated learning in simulation.')

    parser = add_filepath_arguments(parser)
    parser = add_seed_arguments(parser)
    parser = add_algorithm_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_model_arguments(parser)
    parser = add_asr_arguments(parser)
    arguments = parser.parse_args()

    # Initialize random seed
    np.random.seed(arguments.seed)
    torch.random.manual_seed(arguments.seed)

    # Create the local and central privacy mechanisms.
    local_privacy = parse_mechanism(
        mechanism_name=arguments.local_privacy_mechanism,
        clipping_bound=arguments.local_privacy_clipping_bound,
        epsilon=arguments.local_epsilon,
        delta=arguments.local_delta,
        order=arguments.local_order)
    central_privacy = parse_mechanism(
        mechanism_name=arguments.central_privacy_mechanism,
        clipping_bound=arguments.central_privacy_clipping_bound,
        epsilon=arguments.central_epsilon,
        delta=arguments.central_delta,
        order=arguments.central_order,
        cohort_size=arguments.cohort_size,
        noise_cohort_size=arguments.noise_cohort_size,
        num_epochs=arguments.central_num_iterations,
        population=arguments.population,
        min_separation=arguments.min_separation)

    # Create the trie to tokenize the transcripts.
    logger.info('Constructing the trie')
    trie = construct_eng_char_trie_for_ctc(arguments.additional_chars)


    # Create federated training and a central val dataset. Validation federated
    # dataset is not currently used.
    logger.info('Preparing the datasets')
    training_federated_dataset, central_data = get_datasets(arguments, tokenizer=trie, stored_datasets=None)
    val_federated_dataset = None


if __name__ == '__main__':
    main()
