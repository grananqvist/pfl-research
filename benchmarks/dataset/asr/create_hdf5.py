import argparse
import logging
import os.path
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import MinimizeReuseUserSampler
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.argument_parsing import store_bool

from .common import ASRDataset, get_latin_characters, construct_char_trie_for_ctc

logger = logging.getLogger(name=__name__)


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


def create_federated(dataset_type, federated_split, arguments):
    characters = get_latin_characters('en')
    trie = construct_char_trie_for_ctc(characters)

    dataset = ASRDataset(dataset=dataset_type,
                         data_path=arguments.data_path,
                         split=federated_split,
                         trie=trie,
                         target_pad=False,
                         n_threads=arguments.num_threads,
                         stored_datasets=None,
                         dynamic_batching=True, # irrelevant in this context
                         max_sample_audio_length=arguments.max_sample_audio_length,
                         characters=characters,
                         lazy_load_audio=arguments.lazy_load_audio)
    user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    import h5py
    output_file = os.path.join(arguments.data_path, f'federated-{federated_split}.hdf5')
    print('output_file:', output_file)
    with h5py.File(output_file, 'w') as h5file:
        dt = h5py.string_dtype(encoding='utf-8')
        characters_dset = h5file.create_dataset(f'/{federated_split}/metadata', (len(characters) + 1,) , dtype=dt)
        characters_dset.attrs['characters'] = ''.join(sorted(characters))
        for user_id in tqdm(user_ids):
            user_dataset = dataset.make_dataset_fn(user_id)
            df = pd.DataFrame(user_dataset.raw_data)
            h5file[f'/{federated_split}/{user_id}/user_id'] = df['user_id'].to_numpy(dtype=h5py.special_dtype(vlen=np.byte))
            h5file[f'/{federated_split}/{user_id}/input_length'] = df['input_length'].to_numpy(dtype=np.int64)
            h5file[f'/{federated_split}/{user_id}/sum_input_length'] = df['input_length'].sum()
            h5file[f'/{federated_split}/{user_id}/target'] = df['target'].to_numpy(dtype=h5py.special_dtype(vlen=np.int64))
            h5file[f'/{federated_split}/{user_id}/target_length'] = df['target_length'].to_numpy(dtype=np.int64)
            # TODO: We can delete this if we don't plan to use the transcript later on anymore? int8 or byte?
            h5file[f'/{federated_split}/{user_id}/transcript'] = df['transcript'].to_numpy(dtype=h5py.special_dtype(vlen=np.int8))
            h5file[f'/{federated_split}/{user_id}/input_shapes'] = np.stack(df['input'].apply(lambda x: x.shape)) #dtype=h5py.special_dtype(vlen=np.float32))
            h5file[f'/{federated_split}/{user_id}/input'] = df['input'].apply(lambda x: x.flatten()).to_numpy(dtype=h5py.special_dtype(vlen=np.float32)) #dtype=h5py.special_dtype(vlen=np.float32))

def create_federated_ls(federated_split, arguments):
    characters = get_latin_characters('en')
    trie = construct_char_trie_for_ctc(characters)

    dataset = ASRDataset(dataset='librispeech',
                         data_path=arguments.data_path,
                         split=federated_split,
                         trie=trie,
                         target_pad=False,
                         n_threads=arguments.num_threads,
                         stored_datasets=None,
                         dynamic_batching=True, # irrelevant in this context
                         max_sample_audio_length=arguments.max_sample_audio_length,
                         characters=characters)
    user_ids = dataset.get_user_ids()
    print(f'total {len(user_ids)} users')
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    training_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn,
                                        user_sampler=user_sampler)

    import h5py
    output_file = os.path.join(arguments.data_path, f'federated-{federated_split}.hdf5')
    print('output_file:', output_file)
    with h5py.File(output_file, 'w') as h5file:
        for user_id in tqdm(user_ids):
            user_dataset = dataset.make_dataset_fn(user_id)
            df = pd.DataFrame(user_dataset.raw_data)
            h5file[f'/{federated_split}/{user_id}/user_id'] = df['user_id'].to_numpy(dtype=h5py.special_dtype(vlen=np.byte))
            h5file[f'/{federated_split}/{user_id}/input_length'] = df['input_length'].to_numpy(dtype=np.int64)
            h5file[f'/{federated_split}/{user_id}/target'] = df['target'].to_numpy(dtype=h5py.special_dtype(vlen=np.int64))
            h5file[f'/{federated_split}/{user_id}/target_length'] = df['target_length'].to_numpy(dtype=np.int64)
            # TODO: We can delete this if we don't plan to use the transcript later on anymore.
            h5file[f'/{federated_split}/{user_id}/transcript'] = df['transcript'].to_numpy(dtype=h5py.special_dtype(vlen=np.int8))
            h5file[f'/{federated_split}/{user_id}/input_shapes'] = np.stack(df['input'].apply(lambda x: x.shape))
            h5file[f'/{federated_split}/{user_id}/input'] = df['input'].apply(lambda x: x.flatten()).to_numpy(dtype=h5py.special_dtype(vlen=np.float32))


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument('--data_path',
                        default=os.path.expanduser('~/.cache/mlx.data/librispeech'),
                        help='The path from which the dataset will be read.')
    argument_parser.add_argument(
        '--dataset',
        choices=['librispeech', 'cv-en-v13'],
        default="librispeech",
        help='Which dataset to preprocess.')
    argument_parser.add_argument(
        '--federated_splits',
        nargs='+',
        default=['train-clean-100', 'train-clean-360', 'train-other-500'],
        help='Data splits for federated datasets.')
    argument_parser.add_argument("--max_sample_audio_length",
                                 type=int,
                                 default=None,
                                 help='Maximum length of audio for a sample '
                                      'to be used in training or evaluation. '
                                      'All samples with longer audio will be '
                                      'filtered out.')
    argument_parser.add_argument("--num_threads",
                                 type=int,
                                 default=24,
                                 help="Number of threads.")
    argument_parser.add_argument("--lazy_load_audio",
                                 action=store_bool,
                                 default=False,
                                 help='Whether to load audio only once the user dataset or '
                                      'central dataset is created. This typically saves '
                                      'memory for large federated datasets but possibly '
                                      'prolongs the cohort processing.')
    arguments = argument_parser.parse_args()

    for federated_split in arguments.federated_splits:
        create_federated(arguments.dataset, federated_split, arguments)
