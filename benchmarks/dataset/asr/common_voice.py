import logging
from typing import Any, Callable, Dict, List, Optional

from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler

from .common import ASRDataset

logger = logging.getLogger(name=__name__)


def make_cv_datasets(data_path: str,
                     lang: str,
                     training_split: str,
                     validation_split: str,
                     evaluation_splits: List[str],
                     trie: Any,
                     dynamic_batching: bool,
                     stored_datasets: Optional[Dict],
                     target_pad: bool = False):
    logger.info(
        f'Going to preprocess split {training_split} of common-voice dataset (dynamic batching: {dynamic_batching})'
    )
    dataset = ASRDataset(dataset=f'cv-{lang}-v13',
                         data_path=data_path,
                         split=training_split,
                         trie=trie,
                         target_pad=target_pad,
                         n_threads=4,
                         stored_datasets=stored_datasets,
                         dynamic_batching=dynamic_batching)
    user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    logger.info(
        f'Going to preprocess split {validation_split} of common-voice dataset (dynamic batching: {dynamic_batching})'
    )
    dataset = ASRDataset(dataset=f'cv-{lang}-v13',
                         data_path=data_path,
                         split=validation_split,
                         trie=trie,
                         target_pad=target_pad,
                         n_threads=4,
                         stored_datasets=stored_datasets,
                         dynamic_batching=dynamic_batching)
    val_user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_val_dataset_fn = dataset.make_dataset_fn
    val_user_sampler = MinimizeReuseUserSampler(val_user_ids)
    val_dataset = FederatedDataset(make_dataset_fn=make_val_dataset_fn,
                                   user_sampler=val_user_sampler)

    central_data = []
    for split in evaluation_splits:
        logger.info(
            f'Going to preprocess split {split} of common-voice dataset (dynamic batching: {dynamic_batching})')
        dataset = ASRDataset(dataset=f'cv-{lang}-v13',
                             data_path=data_path,
                             split=split,
                             trie=trie,
                             target_pad=target_pad,
                             n_threads=4,
                             stored_datasets=stored_datasets,
                             dynamic_batching=dynamic_batching)
        central_data.append(dataset.full_dataset())

    metadata = {
        'evaluation_splits': evaluation_splits,
    }

    return training_dataset, val_dataset, central_data, metadata
