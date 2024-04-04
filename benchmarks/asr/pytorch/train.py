import argparse
import logging
import os
from uuid import uuid4

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
)
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
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.metrics import StringMetricName
from pfl.model.pytorch import PyTorchModel
from pfl.privacy import CentrallyAppliedPrivacyMechanism

from ..argument_parsing import add_asr_arguments, get_central_lr_schedular
from ..utils import BoltCallback


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

    # Create federated training and a central val dataset. Validation federated
    # dataset is not currently used.
    logger.info('Preparing the datasets')
    training_federated_dataset, val_federated_dataset, central_data, metadata = get_datasets(
        arguments, stored_datasets=None)

    # Trie depends on the dataset so we create it in get_datasets and pass in metadata
    trie = metadata['trie']

    pytorch_model = get_model_pytorch(arguments, num_trie_labels=trie.num_keys())

    params = [p for p in pytorch_model.parameters() if p.requires_grad]
    if arguments.central_optimizer == 'adam':
        # Hyperparameters for stability, see S. Reddi et al. 2020 Appendix C.1.
        central_optimizer = torch.optim.Adam(params,
                                             arguments.learning_rate,
                                             eps=arguments.adaptivity_degree,
                                             betas=(0.9, 0.99))
    elif arguments.central_optimizer == 'lamb':
        import torch_optimizer
        central_optimizer = torch_optimizer.Lamb(params,
                                                 lr=arguments.learning_rate,
                                                 eps=arguments.adaptivity_degree,
                                                 betas=(0.9, 0.99))
    elif arguments.central_optimizer == 'lars':
        from torchlars import LARS
        base_optimizer = torch.optim.SGD(params, arguments.learning_rate)
        central_optimizer = LARS(base_optimizer)
    else:
        assert arguments.central_optimizer == 'sgd'
        central_optimizer = torch.optim.SGD(params, arguments.learning_rate)

    central_lr_schedular = get_central_lr_schedular(arguments, central_optimizer)

    model = PyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=central_optimizer,
                         central_learning_rate_scheduler=central_lr_schedular)

    postprocessors = [
        local_privacy,
        CentrallyAppliedPrivacyMechanism(central_privacy)
    ]

    # TODO: Fix
    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=postprocessors)

    algorithm, algorithm_params, algorithm_callbacks = get_algorithm(arguments)

    model_train_params = NNTrainHyperParams(
        local_learning_rate=arguments.local_learning_rate,
        local_num_epochs=None if arguments.local_num_steps else arguments.local_num_epochs,
        local_num_steps=arguments.local_num_steps,
        local_batch_size=arguments.local_batch_size,
        local_max_grad_norm=arguments.local_max_grad_norm)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=arguments.central_eval_batch_size)

    evaluation_callbacks = []
    for split, central_data_dataset in central_data.items():
        evaluation_callbacks.append(
            CentralEvaluationCallback(central_data_dataset,
                                      model_eval_params=model_eval_params,
                                      frequency=arguments.evaluation_frequency,
                                      format_fn=lambda n, split=split:
                                      StringMetricName(f"{split} | {n}")))

    callbacks = [
        StopwatchCallback(), *evaluation_callbacks,
        AggregateMetricsToDisk(arguments.metrics_file_name),
        # TODO: Add WER and TER here to track when ready.
        # TrackBestOverallMetrics(
        #     lower_is_better_metric_names=['Central val | perplexity']),
    ]

    if arguments.restore_model_path is not None:
        model.load(arguments.restore_model_path)
        logger.info(f'Restored model from {arguments.restore_model_path}')

    callbacks.extend(algorithm_callbacks)

    if arguments.save_model_path is not None:
        callbacks.append(ModelCheckpointingCallback(arguments.save_model_path))

    if arguments.wandb_project_id:
        wandb_dir = os.path.dirname(arguments.metrics_file_name)
        logger.info(f'wandb_dir: {wandb_dir}')
        callbacks.append(
            WandbCallback(
                wandb_project_id=arguments.wandb_project_id,
                wandb_experiment_name=os.environ.get('WANDB_TASK_ID',
                                                     str(uuid4())),
                # List of dicts to one dict.
                wandb_config=dict(vars(arguments)),
                tags=os.environ.get('WANDB_TAGS', 'empty-tag').split(','),
                group=os.environ.get('WANDB_GROUP', None),
                dir=wandb_dir))
    callbacks.append(BoltCallback())

    model = algorithm.run(algorithm_params=algorithm_params,
                          backend=backend,
                          model=model,
                          model_train_params=model_train_params,
                          model_eval_params=model_eval_params,
                          callbacks=callbacks)


if __name__ == '__main__':
    main()
