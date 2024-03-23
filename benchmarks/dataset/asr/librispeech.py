import logging
from typing import Any, Callable, Dict, List, Optional

from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler

from .common import ASRDataset, get_latin_characters, construct_char_trie_for_ctc

logger = logging.getLogger(name=__name__)


def make_librispeech_datasets(data_path: str,
                              training_split: str,
                              validation_split: str,
                              evaluation_splits: List[str],
                              dynamic_batching: bool,
                              stored_datasets: Optional[Dict],
                              target_pad: bool = False,
                              max_sample_audio_length: Optional[int] = None,
                              num_threads: int = 1,
                              lazy_load_audio: bool = False):
    logger.info(
        f'Going to preprocess split {training_split} of librispeech dataset (dynamic batching: {dynamic_batching})'
    )
    characters = get_latin_characters('en')
    trie = construct_char_trie_for_ctc(characters)

    dataset = ASRDataset(dataset='librispeech',
                         data_path=data_path,
                         split=training_split,
                         trie=trie,
                         target_pad=target_pad,
                         n_threads=num_threads,
                         stored_datasets=stored_datasets,
                         dynamic_batching=dynamic_batching,
                         max_sample_audio_length=max_sample_audio_length,
                         characters=characters,
                         lazy_load_audio=lazy_load_audio)
    user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    logger.info(
        f'Going to preprocess split {validation_split} of librispeech dataset (dynamic batching: {dynamic_batching})'
    )
    dataset = ASRDataset(dataset='librispeech',
                         data_path=data_path,
                         split=validation_split,
                         trie=trie,
                         target_pad=target_pad,
                         n_threads=num_threads,
                         stored_datasets=stored_datasets,
                         dynamic_batching=dynamic_batching,
                         max_sample_audio_length=max_sample_audio_length,
                         characters=characters,
                         lazy_load_audio=lazy_load_audio)
    val_user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_val_dataset_fn = dataset.make_dataset_fn
    val_user_sampler = MinimizeReuseUserSampler(val_user_ids)
    val_dataset = FederatedDataset(make_dataset_fn=make_val_dataset_fn,
                                   user_sampler=val_user_sampler)

    central_data = {}
    for split in evaluation_splits:
        logger.info(
            f'Going to preprocess split {split} of librispeech dataset')
        dataset = ASRDataset(dataset='librispeech',
                             data_path=data_path,
                             split=split,
                             trie=trie,
                             target_pad=target_pad,
                             n_threads=num_threads,
                             stored_datasets=stored_datasets,
                             dynamic_batching=dynamic_batching,
                             max_sample_audio_length=max_sample_audio_length,
                             characters=characters,
                             lazy_load_audio=lazy_load_audio)
        central_data[split] = dataset.full_dataset()

    metadata = {
        'characters': characters,
        'trie': trie,
    }

    return training_dataset, val_dataset, central_data, metadata
