import h5py
import argparse
import os
import time
import numpy as np
import mlx.data as dx
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler
from .common import MlxDataUserDataset, construct_char_trie_for_ctc
import logging


logger = logging.getLogger(name=__name__)


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


def get_central_dataset(data_path: str, split: str):
    hdf5_path = os.path.join(data_path, f'federated-{split}.hdf5')

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        split = list(hdf5_file.keys())[0]
        logger.info(f'Loading central dataset from {hdf5_path}, split {split}')
        user_ids = list(hdf5_file[split].keys())
        characters = hdf5_file[split]['metadata'].attrs['characters']
        print(f'\ttotal {len(user_ids)} users')
        print(f'\tcharacters: "{characters}"')
        trie = construct_char_trie_for_ctc(characters)
        all_data = process_users(h5file, split, user_ids)
        dataset = dx.buffer_from_vector(list(all_data))
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

    argument_parser.add_argument('--data_path',
                                 type=str,
                                 default=os.path.expanduser('~/.cache/mlx.data/cv-v13'),
                                 help='The data path from which the datasets will be read.')
    argument_parser.add_argument('--split',
                                 default='en-train',
                                 help='The split.')

    arguments = argument_parser.parse_args()

    dataset = get_central_dataset(arguments.data_path, arguments.split)
    # print('Creating central dataset')
    # central_dataset = make_central_dataset(user_ids, hdf5_path=arguments.hdf5_path, split=split, dynamic_batching_key='input', trie=trie)

