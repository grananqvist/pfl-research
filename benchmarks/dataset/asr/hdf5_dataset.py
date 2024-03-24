import h5py
import argparse
import logging
import os
import time
import numpy as np
import mlx.data as dx
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler
from .common import MlxDataUserDataset, construct_char_trie_for_ctc
from typing import List


logger = logging.getLogger(name=__name__)


def make_datasets(data_path: str,
                  training_split: str,
                  validation_split: str,
                  evaluation_splits: List[str],
                  dynamic_batching: bool):
    training_dataset, characters, trie = get_federated_dataset(
        data_path, training_split, dynamic_batching)
    if validation_split:
        val_dataset, new_characters, _ = get_federated_dataset(
            data_path, validation_split, dynamic_batching)
        assert new_characters == characters
    else:
        val_dataset = None
    central_data = {}
    for evaluation_split in evaluation_splits:
        central_data[evaluation_split], new_characters, _ = get_central_dataset(
            data_path, evaluation_split, dynamic_batching)
        assert new_characters == characters

    metadata = {
        'characters': characters,
        'trie': trie,
    }

    return training_dataset, val_dataset, central_data, metadata


def process_user(h5file, split, user_id, quiet=True):
    user_datasets = h5file[split][user_id]
    # print('user_datasets:', user_datasets)
    # print('user_datasets.keys():', user_datasets.keys())
    input = np.array(h5file[split][user_id]['input'])
    input_length = np.array(h5file[split][user_id]['input_length'])
    input_shapes = np.array(h5file[split][user_id]['input_shapes'])
    target = np.array(h5file[split][user_id]['target'])
    target_length = np.array(h5file[split][user_id]['target_length'])
    transcript = np.array(h5file[split][user_id]['transcript'])
    user_id = np.array(h5file[split][user_id]['user_id'])


    # print('input:', input)
    input = [x.reshape(x_shape) for (x, x_shape) in zip(input, input_shapes)]
    for vals in zip(input, input_length, target, target_length, transcript, user_id):
        keys = ['input', 'input_length', 'target', 'target_length', 'transcript', 'user_id']
        yield dict(zip(keys, vals))


def make_user_dataset(user_id, hdf5_path, split, dynamic_batching_key, trie):
    with h5py.File(hdf5_path, 'r') as h5file:
        user_data = process_user(h5file, split, user_id)
        dataset = dx.buffer_from_vector(list(user_data))
        # features = np.array(h5[f'/{user_id}/features'])
        # labels = np.array(h5[f'/{user_id}/labels'])
        # # Randomly shuffle the user dataset each time when loading
        # data_order = np.random.permutation(len(features))
        return MlxDataUserDataset(
            dataset,
            user_id=user_id,
            dynamic_batching_key=dynamic_batching_key,
            trie=trie)


def get_central_dataset(data_path: str, split: str, dynamic_batching: bool):
    def process_users(h5file, split, user_ids):
        for user_id in user_ids:
            yield from process_user(h5file, split, user_id)

    hdf5_path = os.path.join(data_path, f'federated-{split}.hdf5')

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        split = list(hdf5_file.keys())[0]
        logger.info(f'Loading central dataset from {hdf5_path}, split {split}')
        user_ids = [key for key in hdf5_file[split].keys() if key != 'metadata']
        characters = hdf5_file[split]['metadata'].attrs['characters']
        print(f'\ttotal {len(user_ids)} users')
        print(f'\tcharacters: "{characters}"')
        trie = construct_char_trie_for_ctc(characters)
        all_data = process_users(hdf5_file, split, user_ids)
        dataset = dx.buffer_from_vector(list(all_data))
        dynamic_batching_key = 'input' if dynamic_batching else None
        return MlxDataUserDataset(
            dataset,
            user_id=None,
            dynamic_batching_key=dynamic_batching_key,
            trie=trie), characters, trie



def get_federated_dataset(data_path: str, split: str, dynamic_batching: bool):
    def make_user_dataset(user_id, hdf5_path, split, dynamic_batching_key, trie):
        with h5py.File(hdf5_path, 'r') as h5file:
            user_data = process_user(h5file, split, user_id)
            dataset = dx.buffer_from_vector(list(user_data))
        return MlxDataUserDataset(
            dataset,
            user_id=user_id,
            dynamic_batching_key=dynamic_batching_key,
            trie=trie)

    hdf5_path = os.path.join(data_path, f'federated-{split}.hdf5')

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        split = list(hdf5_file.keys())[0]
        logger.info(f'Loading central dataset from {hdf5_path}, split {split}')
        user_ids = [key for key in hdf5_file[split].keys() if key != 'metadata']
        user_id_to_weight = {
            user_id: np.int32(hdf5_file[split][user_id]['sum_input_length']) for user_id in user_ids
        }
        # print('user_id_to_weight:', user_id_to_weight)
        characters = hdf5_file[split]['metadata'].attrs['characters']
        print(f'\ttotal {len(user_ids)} users')
        print(f'\tcharacters: "{characters}"')
        trie = construct_char_trie_for_ctc(characters)

        dynamic_batching_key = 'input' if dynamic_batching else None
        make_dataset_fn = (lambda user_id, hdf5_path=hdf5_path, split=split, dynamic_batching_key=dynamic_batching_key, trie=trie: \
            make_user_dataset(user_id, hdf5_path, split, dynamic_batching_key, trie))
        user_sampler = MinimizeReuseUserSampler(user_ids)
        dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                   user_sampler=user_sampler,
                                   user_id_to_weight=user_id_to_weight)
        return dataset, characters, trie

    raise Exception(f'Could not proccess the split {split} in {data_path}, i.e. the file {hdf5_path}')
