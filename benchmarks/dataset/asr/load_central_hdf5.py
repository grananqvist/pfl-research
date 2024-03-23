import h5py
import argparse
import os
import time
import numpy as np
import mlx.data as dx
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler
from .common import MlxDataUserDataset


processing_times = []


def process_user(h5file, split, user_id, quiet=True):
    start = time.time()
    user_datasets = h5file[split][user_id]
    input = np.array(h5file[split][user_id]['input'])
    input_length = np.array(h5file[split][user_id]['input_length'])
    input_shapes = np.array(h5file[split][user_id]['input_shapes'])
    target = np.array(h5file[split][user_id]['target'])
    transcript = np.array(h5file[split][user_id]['transcript'])
    user_id = np.array(h5file[split][user_id]['user_id'])


    # print('input:', input)
    input = [x.reshape(x_shape) for (x, x_shape) in zip(input, input_shapes)]
    end = time.time()
    processing_times.append(end-start)
    if not quiet:
        print(f'processed user in {end - start} seconds')
    for vals in zip(input, input_length, target, transcript, user_id):
        keys = ['input', 'input_length', 'target', 'transcript', 'user_id']
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


def make_central_dataset(user_ids, hdf5_path, split, dynamic_batching_key, trie):
    with h5py.File(hdf5_path, 'r') as h5file:
        all_data = process_users(h5file, split, user_ids)
        dataset = dx.buffer_from_vector(list(all_data))
        # features = np.array(h5[f'/{user_id}/features'])
        # labels = np.array(h5[f'/{user_id}/labels'])
        # # Randomly shuffle the user dataset each time when loading
        # data_order = np.random.permutation(len(features))
    return MlxDataUserDataset(
        dataset,
        user_id=None,
        dynamic_batching_key=dynamic_batching_key,
        trie=trie)


def process_users(h5file, split, user_ids):
    for user_id in user_ids:
        yield from process_user(h5file, split, user_id)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument('--hdf5_path',
                        default=os.path.expanduser('~/.cache/mlx.data/cv-v13/federated-en-dev.hdf5'),
                        help='The hdf5 file from which the dataset will be read.')

    arguments = argument_parser.parse_args()

    from .common import construct_char_trie_for_ctc, get_latin_characters
    characters = get_latin_characters('en') # TODO: Do this via h5 file
    trie = construct_char_trie_for_ctc(characters)

    with h5py.File(arguments.hdf5_path, 'r') as h5file:
        split = list(h5file.keys())[0]
        user_ids = list(h5file[split].keys())
        print(f'total {len(user_ids)} users')

    print('Creating federated dataset')
    make_dataset_fn = lambda user_id, hdf5_path=arguments.hdf5_path, split=split, dynamic_batching_key='input', trie=trie: make_user_dataset(user_id, hdf5_path, split, dynamic_batching_key, trie)

    user_sampler = MinimizeReuseUserSampler(user_ids)
    dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                               user_sampler=user_sampler)

    for index in range(2000):
        raw_user_data = next(dataset)[0].raw_data
        if index < 5:
            for item in raw_user_data:
                print('item["input"].shape:', item["input"].shape, 'user_id:', bytes(item["user_id"]))
        if index < 20:
            print(f'user dataset size: {raw_user_data.size()}')
        #
        # print(np.array(process_users(h5file, split, user_ids[:2])))
        #
        # buffer = dx.buffer_from_vector(np.array(process_users(h5file, split, user_ids[:2])))
        #
        # print(buffer)

    # print('Creating central dataset')
    # central_dataset = make_central_dataset(user_ids, hdf5_path=arguments.hdf5_path, split=split, dynamic_batching_key='input', trie=trie)

    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    print('total num processed:', len(processing_times))
    print(f'processing times quantiles {q}: {np.quantile(processing_times, q=q)}')
