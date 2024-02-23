from .common import ASRDataset
from typing import Any, Callable, Dict, List, Optional
import logging
from pfl.data.sampling import MinimizeReuseUserSampler
from pfl.data.federated_dataset import FederatedDataset


logger = logging.getLogger(name=__name__)


def make_librispeech_datasets(
            data_path: str,
            training_split: str,
            evaluation_splits: List[str],
            tokenizer: Any,
            dynamic_batching: bool,
            stored_datasets: Optional[Dict],
            target_pad: bool = False):
    logger.info(f'Going to preprocess split {training_split} of librispeech dataset')
    dataset = ASRDataset(
        dataset='librispeech',
        data_path=data_path,
        split=training_split,
        trie=tokenizer,
        target_pad=target_pad,
        n_threads=4,
        stored_datasets=stored_datasets,
        dynamic_batching=dynamic_batching)
    user_ids = dataset.get_user_ids()
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    central_data = []
    for split in evaluation_splits:
        logger.info(f'Going to preprocess split {split} of librispeech dataset')
        training_dataset = ASRDataset(
            dataset='librispeech',
            data_path=data_path,
            split=split,
            trie=tokenizer,
            target_pad=target_pad,
            n_threads=4,
            stored_datasets=stored_datasets,
            dynamic_batching=dynamic_batching)
        central_data.append(dataset.full_dataset())

    return training_dataset, central_data
