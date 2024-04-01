"""
Script for preprocessing common-voice from mozilla or librispeech
from openslr.org.

For common-voice, the script processes the data in two stages:
1. Converting and downsampling of relevant mp3 files to flac.
2. Processing the flac data to generate hdf5 file with all necessary
   data for training and evaluation with pfl-research. This includes
   tokenization and other transformations.

For librispeech, only stage 2 is necessary; the processing is faster
and requires less disk space.

To run this file, you must first download the common-voice data
from https://commonvoice.mozilla.org/en/datasets
or librispeech data from http://openslr.org/12/

"""
import argparse
import pandas as pd
import sox
from joblib import Parallel, delayed
import csv
import os
import mlx.data.datasets as datasets
from .common import ASRDataset, construct_char_trie_for_ctc, get_latin_characters, process_latin_sentence
from typing import List, Optional
from utils.argument_parsing import store_bool
import re
from collections import defaultdict
import tarfile
from pfl.data.sampling import MinimizeReuseUserSampler
from pfl.data.federated_dataset import FederatedDataset
from tqdm import tqdm
import numpy as np

def convert_mp3_to_flac(input_file, sampling_rate=16000):
    output_file = input_file.replace(".mp3", ".flac").replace("clips", "flac-16kHz")
    sox_tfm = sox.Transformer()
    sox_tfm.set_output_format(file_type="flac", encoding="signed-integer", rate=sampling_rate, bits=16)
    sox_tfm.build(input_file, output_file)


def process_audio_files(cv_base_path, languages, num_threads):
    for language in languages:
        all_files = []
        cv_lang_path = os.path.join(cv_base_path, language)
        for name in ["train.tsv", "dev.tsv", "test.tsv"]:
            print(f'Reading {cv_lang_path}/{name}')
            data = pd.read_csv(os.path.join(cv_lang_path, name), sep="\t", engine=None, encoding="utf8", quoting=csv.QUOTE_NONE)
            all_files += list(data["path"])

        os.mkdir(os.path.join(cv_lang_path, "flac-16kHz"))
        all_files = [os.path.join(os.path.join(cv_lang_path, "clips"), i) for i in all_files]

        print(f'Going to process {len(all_files)} audio files for language {language} with {num_threads} threads')
        with Parallel(n_jobs=num_threads) as parallel:
            parallel(delayed(convert_mp3_to_flac)(input_file, 16000) for input_file in all_files)


def preprocess_latin_text(cv_base_path: str, languages: List[str], default_characters: Optional[str] = None):
    for language in languages:
        characters = default_characters if default_characters else get_latin_characters(language)
        characters_without_punctuation = characters - set("-'")
        disallowed_punctuation = set("!\"#$%&\()*+,./:;<=>?@[\\]^_`{|}~")
        all_characters = defaultdict(int)
        cv_lang_path = os.path.join(cv_base_path, language)

        for name in ["train.tsv", "dev.tsv", "test.tsv"]:
            print(f'Going to process text for language {language} in {os.path.join(cv_lang_path, name)}')
            data = pd.read_csv(os.path.join(cv_lang_path, name), sep="\t", quoting=csv.QUOTE_NONE)
            data_new = dict()
            data_new["filename"] = [p.replace("mp3", "flac").replace("clips", "flac-16kHz") for p in list(data["path"])]
            data_new = pd.DataFrame(data_new)
            data_new["id"] = data_new["filename"]
            data_new["transcription"] = [process_latin_sentence(str(tr), characters, characters_without_punctuation, disallowed_punctuation, all_characters) for tr in data["sentence"]]
            data_new["raw_transcription"] = data["sentence"]
            data_new["client_id"] = data["client_id"]
            # data_new["tar_file"] = cv_base_path.split("/")[-1] + ".tar"
            print("\tdata len", len(data_new))
            data_new = data_new.dropna()
            print("\tdata len after dropna", len(data_new))
            data_new.to_csv(os.path.join(cv_lang_path, cv_lang_path.split("/")[-1] + "-" + name), sep="\t", index=False)

        s = b""
        for k, v in all_characters.items():
            if k.decode() in characters:
                continue
            s += k
        print('Done processing the text files for all languages.')
        assert len(s) == 0


def create_federated(dataset_type, federated_split, arguments):
    characters = get_latin_characters('en')  # TODO: Must change to handle other languages
    print('characters:', characters)
    trie = construct_char_trie_for_ctc(characters)

    dataset = ASRDataset(dataset=dataset_type,
                         data_path=arguments.data_path,
                         split=federated_split,
                         trie=trie,
                         target_pad=False,
                         n_threads=arguments.num_threads,
                         stored_datasets=None,
                         dynamic_batching=True, # irrelevant in this context
                         max_sample_audio_length=None,
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


def preprocess_common_voice_audio(arguments):
    pwd = os.getcwd()

    cv_archive_path = os.path.join(arguments.data_path, 'cv-corpus-13.0-2023-03-09-en.tar.gz')
    cv_base_path = None

    assert os.path.isfile(cv_archive_path)
    tar = None
    language = None
    if cv_archive_path.endswith("tar.gz"):
        tar = tarfile.open(cv_archive_path, "r:gz")
    elif cv_archive_path.endswith("tar"):
        tar = tarfile.open(cv_archive_path, "r:")

    if tar:
        print(f'Unpacking {cv_archive_path}')
        first_filename = tar.next().name.split('/')
        language = first_filename[1] # We assume each tar file contains one language.
        if cv_base_path is None:
            cv_base_path = os.path.join(os.path.dirname(cv_archive_path), first_filename[0])
        else:
            assert cv_base_path == os.path.join(os.path.dirname(cv_archive_path), first_filename[0])
        # TODO: We can use tar directly (but it is slower with more threads)
        tar.extractall(path=os.path.dirname(cv_archive_path))
        tar.close()
    else:
        raise Exception('Unsupported archive type')

    print('cv_base_path:', cv_base_path)
    preprocess_latin_text(cv_base_path, [language])
    process_audio_files(cv_base_path, [language], arguments.num_threads)

    def tar_filter_fn(x):
        if x.isdir():
            if x.name.endswith('clips'):
                return None
            return x
        elif x.name.endswith('.flac'):
            return x
        elif x.name.endswith('.tsv') and os.path.basename(x.name).startswith(f'{language}-'):
            return x

    os.chdir(os.path.dirname(cv_base_path))
    data_path = cv_base_path.split('/')[-1]
    output_tar_filenames = []
    output_tar_filename = f'{cv_base_path}-flac-{language}.tar'
    output_tar_filenames.append(output_tar_filename)
    print(f'Compressing {output_tar_filename}')
    with tarfile.open(output_tar_filename, 'w') as tar:
        tar.add(os.path.join(data_path, language),
                recursive=True,
                filter=tar_filter_fn)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument(
        '--dataset',
        type=str,
        default='librispeech',
        choices=['librispeech', 'cv-en-v13'],
        help='Dataset type. This is necessary, since the data from different '
             'sources must be preprocessed differently.')

    argument_parser.add_argument(
        '--splits',
        nargs='+',
        default=None,
        help='Splits to process, e.g. ["train", "dev", "test"] for cv-en-v13. '
             'If unset, the splits are determined automatically depending on '
             'the dataset type.')

    argument_parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join(datasets.common.CACHE_DIR, 'librispeech'),
        help='Path where to find the downloaded tar or tar.gz archive. '
             'The filenames are determined from the dataset type.')

    argument_parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path where to save output files.')

    argument_parser.add_argument(
        '--num_threads',
        type=int,
        default=30,
        help="Number of threads. Set to number of cpus if 0.")

    argument_parser.add_argument(
        "--lazy_load_audio",
        action=store_bool,
        default=False,
        help='Whether to load audio only once the user dataset is '
             'created. This typically saves memory for large '
             'federated datasets but makes the processing much longer.')

    arguments = argument_parser.parse_args()

    if arguments.dataset == 'cv-en-v13':
        preprocess_common_voice_audio(arguments)

    if arguments.splits:
        splits = arguments.splits
    elif arguments.dataset == 'librispeech':
        splits = ['train-clean-100', 'train-clean-360', 'train-other-500',
                  'dev-clean', 'dev-other', 'test-clean', 'test-other']
    elif arguments.dataset == 'cv-en-v13':
        splits = ['train', 'dev', 'test']
    else:
        raise ValueError(f'Unknown dataset type {arguments.dataset}')

    for split in splits:
        print(f'Processing split {split}')
        create_federated(arguments.dataset, split, arguments)
