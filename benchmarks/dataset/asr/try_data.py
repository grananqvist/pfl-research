import logging
from typing import Any, Callable, Dict, List, Optional

from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler

from .common import ASRDataset
import os
from mlx.data.core import CharTrie
import numpy as np


def construct_eng_char_trie_for_ctc(additional_chars):
    trie = CharTrie()
    trie.insert("@")  # blank
    trie.insert(" ")
    trie.insert("'")
    for c in range(ord("a"), ord("z") + 1):
        trie.insert(chr(c))
    if additional_chars:
        for c in additional_chars:
            trie.insert(c)
    return trie


def main():
    logger = logging.getLogger(name=__name__)

    lang='en'
    data_path=os.path.expanduser('~/.cache/mlx.data/cv-en-v13')
    training_split='dev'
    trie=construct_eng_char_trie_for_ctc('')
    target_pad=False
    num_threads=1
    stored_datasets=None
    dynamic_batching=True
    max_sample_audio_length=384000
    batch_size=384000
    logger.info(
        f'Going to preprocess split {training_split} of common-voice dataset (dynamic batching: {dynamic_batching})'
    )
    dataset = ASRDataset(dataset=f'cv-{lang}-v13',
                         data_path=data_path,
                         split=training_split,
                         trie=trie,
                         target_pad=target_pad,
                         n_threads=num_threads,
                         stored_datasets=stored_datasets,
                         dynamic_batching=dynamic_batching,
                         max_sample_audio_length=max_sample_audio_length)
    user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    cohort = training_dataset.get_cohort(5)
    for user_dataset, _ in cohort:
        # Print shapes of first 3 static batches of size 5 for this user
        # These would be sent to the model for calculating the loss.
        print(
            f"\tReal user {user_dataset.user_id} has size of {len(user_dataset)}.")
        for idx, x in enumerate(user_dataset.iter(batch_size)):
            print(
                f"\t\tinput.shape: {x['input'].shape}   target.shape: {x['target'].shape}   total audio: {np.prod(x['input'].shape)}"
            )
            if idx == 2:
                break


if __name__ == '__main__':
    main()
