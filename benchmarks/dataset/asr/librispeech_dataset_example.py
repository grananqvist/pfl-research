#!/usr/bin/env python
# coding: utf-8

# Example of creating a pfl-research federated dataset for librispeech.


import mlx.data as dx
import numpy as np
from mlx.data.core import CharTrie
from pfl.data.sampling import MinimizeReuseUserSampler
from pfl.model.pytorch import PyTorchModel  # pylint: disable=unused-import
#import torch
import pandas as pd
import time
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.dataset import Dataset
import os
from typing import Optional, List, Dict, Any, Tuple


class LibriSpeechDataset:
    def __init__(
            self,
            name,
            prefix,
            trie,
            stored_datasets,  # TODO: Do we only need this for the central dataset?
            policy="sorted",
            max_target_length=400,
            seed=42,
            n_threads=-1,
            target_pad=False,
    ):
        assert trie is not None

        self.policy = policy

        # update stored_datasets
        np.random.seed(seed=seed)
        if name in stored_datasets:
            self.dataset = stored_datasets[name]
        else:
            start = time.time()
            self.dataset = (dx.files_from_tar(name)
                            .to_stream()
                            .sample_transform(lambda s: s if bytes(s["file"]).endswith(b".txt") else dict())
                            .read_from_tar(name, "file", "samples", )
                            .line_reader_from_key("samples", "sample", from_memory=True)
                            .prefetch(n_threads, n_threads) # TODO: check where to prefetch
                            .sample_transform(self.process_csv_row)
                            .prefetch(n_threads, n_threads) # TODO: check where to prefetch
                            # .to_buffer()
                            .read_from_tar(name, "audio_file", "audio", prefix=prefix)
                            .load_audio("audio", from_memory=True, output_key="input")
                            .sample_transform(self.add_duration)
                            .prefetch(n_threads, n_threads) # TODO: check where to prefetch
                            .tokenize("transcript", trie, ignore_unk=True, output_key="target")
                            .to_buffer()
                            .pad_to_multiple("input", 0, 16000, 0)
                            .shape("input", "input_length", 0)
                            .squeeze("input", -1)  # %1s, one channel
                            .pad_to_size("target", 0, max_target_length, 0)
                            # .squeeze("input_length", -1)  # scalar batch # TODO: fix if needed
                            # .squeeze("target_length", -1)  # scalar batch # TODO: fix if needed
                            # .pad_to_multiple("input", 0, num_local_devices, 0)
                            # .pad_to_multiple("target", 0, num_local_devices, 1)  # sil
                            # .pad_to_multiple("input_length", 0, num_local_devices, 0)
                            # .pad_to_multiple("target_length", 0, num_local_devices, 0)
                            )
            if target_pad:
                self.dataset = self.dataset.pad("target", 0, 1, 1, 1)  # pad target with silence

            self.dataset = self.dataset.shape("target", "target_length", 0)

            end = time.time()
            print(f'    time for initializing the dataset buffer: {end - start}')

            stored_datasets[name] = self.dataset

        start = time.time()

        self.durations, self.client_ids = zip(
            *map(lambda item: [item['duration'].item(), item['user_id'].item()], self.dataset))

        end = time.time()
        print(f'    time for extracting durations and client ids: {end - start}')

        # duration_perm = np.argsort(np.array(durations))[rank::num_nodes]
        # durations = np.array(durations)[duration_perm]
        # client_ids = np.array(client_ids)[duration_perm]
        # dataset = dataset.perm(duration_perm)
        self.durations = np.array(self.durations)
        self.client_ids = np.array(self.client_ids)

        print("Dataset total (h) duration",
              np.sum(self.durations) / 16000 / 60 / 60)  # TODO: Check why not exact match for total duration

        start = time.time()

        self.clients_unique = np.unique(self.client_ids)
        print(f'total {len(self.clients_unique)} clients')

        self.df = pd.DataFrame({"duration": self.durations, "client": self.client_ids})
        self.df_group = self.df.groupby("client", sort=False).groups

        end = time.time()
        print(f'    time for grouping and other postprocessing: {end - start}')


    @staticmethod
    def process_csv_row(sample):
        # Split the line
        file_part, transcript = bytes(sample["sample"]).split(b" ", 1)

        # Extract the audio path
        parts = file_part.split(b"-")
        parts[-1] = file_part + b".flac"
        audio_path = b"/".join(parts)

        # Prepare the transcript
        transcript = transcript.lower()

        # User id
        user_id = int(parts[-3])

        return {"audio_file": audio_path, "transcript": transcript, "user_id": user_id}

    @staticmethod
    def add_duration(sample):
        duration = sample['input'].shape[0]

        sample['duration'] = duration
        return sample

    def get_user_ids(self):
        return self.clients_unique

    def get_user_dataset(self, user_id):
        # We will shuffle inside dataset iter before serving batches so do not have to here (for random policy)
        tmp = self.df_group[user_id]
        client_permutation = tmp

        dataset = self.dataset.perm(client_permutation)

        return dataset

    def make_dataset_fn(self, user_id):
        # print('user_id:', user_id)

        dataset = self.get_user_dataset(user_id)
        return UserDatasetASR(dataset, user_id=user_id)


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


trie = construct_eng_char_trie_for_ctc('')
stored_datasets = {} # TODO: We probably don't need this?


class UserDatasetASR(Dataset):
    """
    A representation of a user dataset for ASR based on mlx.data.

    :param raw_data:
        A buffer with preprocessed raw data (these are not padded or
        batched).
    :param user_id:
        (Optional) String user identifier.
    :param metadata:
        (Optional) Store additional data about the user. Can be retrieved
        later by the algorithm.
    :param train_kwargs:
        A dictionary of any additional training parameters to unpack in the
        training call of the deep learning framework.
    :param eval_kwargs:
        A dictionary of any additional evaluation parameters to unpack in the
        evaluation call of the deep learning framework.
    :param shuffle:
        Shuffle the data before producing batches.
    """

    def __init__(self,
                 raw_data: dx.Buffer,
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None,
                 val_indices: Optional[List[int]] = None,
                 shuffle: bool = True):
        self._raw_data = raw_data
        self._durations = np.array([item['input_length'].item() for item in raw_data])
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._val_indices = val_indices
        # Cache batch splits because it is time consuming.
        self._batches: Dict = {}
        self._shuffle = shuffle


    def __len__(self):
        """
        Recursively search for an `np.array` in `raw_data` and return its
        first dimension, which should be the number of data points.
        """
        # return self._first_tensor_length(self.raw_data)
        return self._raw_data.size()


    def iter(self, batch_size: Optional[int]):  # noqa: A003
        if batch_size is None:
            batch_size = self._raw_data.size()
            dataset = self._raw_data.batch(batch_size=batch_size)
            yield dataset
            # yield self.raw_data
            # return


        def get_batch_sizes(dataset, batch_size):
            batch_size_array = []
            current_max_duration = 0
            current_batch_size = 0
            for duration in self._durations:
                new_max = max(current_max_duration, duration)
                # padding to 1s
                # new_max = new_max + (16000 - new_max % 16000) # don't need since we already pad to multiple of 1s
                # padding to hav same number per gpu
                # new_batch = (current_batch_size + 1) + (
                #         num_local_devices - (current_batch_size + 1) % num_local_devices
                # )
                new_batch = current_batch_size + 1
                if new_batch * new_max > batch_size:
                    batch_size_array.append(current_batch_size)
                    current_batch_size = 0
                    current_max_duration = 0
                current_batch_size += 1
                current_max_duration = max(current_max_duration, duration)
            if current_batch_size > 0:
                batch_size_array.append(current_batch_size)

            return batch_size_array


        if batch_size not in self._batches:
            # TODO: Having some rare issues with the mlx.data dynamic batching so using a custom one.
            if self._shuffle:
                random_perm = np.random.permutation(len(self))
                self._durations = np.array(self._durations)[random_perm]
                dataset = self._raw_data.perm(random_perm)
            else:
                dataset = self._raw_data

            batch_size_array = get_batch_sizes(dataset, batch_size)
            dataset = dataset.batch(batch_size_array)

            yield from dataset


    # TODO: Modified but didn't test so far.
    def split(
            self,
            fraction: Optional[float] = None,
            min_train_size: Optional[int] = None,
            min_val_size: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
        if fraction is None:
            fraction = 0.8
        if min_train_size is None:
            min_train_size = 1
        if min_val_size is None:
            min_val_size = 1

        data_length = len(self)
        if min_train_size + min_val_size > data_length:
            raise ValueError(
                f"The dataset is only of size {data_length}. To satisfy "
                "min_train_size and min_val_size, it must be at "
                f"least of size {min_train_size + min_val_size}")

        split_index = int(data_length * fraction)
        # Possibly override initial split to satisfy splits.
        split_index -= max(0, min_val_size - (data_length - split_index))
        split_index += max(0, min_train_size - split_index)
        left_slice = range(0, split_index)
        right_slice = range(split_index, data_length)

        left_raw_data = self._raw_data[left_slice]
        right_raw_data = self._raw_data[right_slice]
        train_dataset = UserDatasetASR(left_raw_data,
                                       user_id=self.user_id,
                                       metadata=self.metadata,
                                       train_kwargs=self.train_kwargs,
                                       eval_kwargs=self.eval_kwargs,
                                       shuffle=self._shuffle)
        val_dataset = UserDatasetASR(right_raw_data,
                                     user_id=self.user_id,
                                     metadata=self.metadata,
                                     train_kwargs=self.train_kwargs,
                                     eval_kwargs=self.eval_kwargs,
                                     shuffle=self._shuffle)
        return train_dataset, val_dataset

    def get_worker_partition(self) -> 'AbstractDataset':
        partition_range = get_ops().distributed.distribute_range(len(self))
        return UserDatasetASR(raw_data=self._raw_data[partition_range],
                              train_kwargs=self.train_kwargs,
                              eval_kwargs=self.eval_kwargs,
                              shuffle=self._shuffle)


def create_federated_dataset(split, trie, stored_datasets):
    dataset = LibriSpeechDataset(
        name=os.path.expanduser(f'~/.cache/mlx.data/librispeech/{split}.tar'),
        prefix=f"LibriSpeech/{split}",
        trie=trie,
        stored_datasets=stored_datasets,
        policy="random",
        max_target_length=400,
        seed=42,
        n_threads=8,
        target_pad=False)
    user_ids = dataset.get_user_ids()
    make_dataset_fn = dataset.make_dataset_fn
    user_sampler = MinimizeReuseUserSampler(user_ids)
    federated_dataset = FederatedDataset(make_dataset_fn=make_dataset_fn, user_sampler=user_sampler)
    return federated_dataset, user_ids


def get_timing(split='train-clean-100', max_runs_over_data=2, cohort_size=10):
    print('\n=============================================================================')
    print('Split:', split)

    trie = construct_eng_char_trie_for_ctc('')

    start = time.time()
    federated_dataset, user_ids = create_federated_dataset(split, trie, stored_datasets)

    end = time.time()
    initial_time = end - start

    print(f'Initial creation of the federated dataset: {initial_time}')

    total_processed = 0
    first_report = len(user_ids)
    max_processed = 2 * len(user_ids)

    start = time.time()
    while total_processed < max_processed:
        for client_dataset, _ in federated_dataset.get_cohort(cohort_size):
            total_processed += 1
            # if total_processed == 1:
            #     print('   ...shapes of 1st user:', [x.shape for x in client_dataset.raw_data.raw_data])
            # if total_processed == 2:
            #     print('   ...shapes of 2nd user:', [x.shape for x in client_dataset.raw_data.raw_data])
            if total_processed == first_report:
                first_report_time = time.time() - start
                print(
                    f'Time until {total_processed} users processed: {first_report_time}  (per-user: {first_report_time / total_processed})')
                last_report = total_processed
            if total_processed >= max_processed:
                break
    if total_processed > last_report:
        last_report_time = time.time() - start
        print(
            f'Time until {total_processed} users processed: {last_report_time}  (per-user: {last_report_time / total_processed})')
    else:
        last_report_time = None

    print(f'{initial_time} initialization')
    print(f'{initial_time + first_report_time} initialization + 1 pass over data')
    if last_report_time:
        print(f'{initial_time + last_report_time} initialization + {max_runs_over_data} passes over data')


stored_datasets = {}
get_timing('dev-clean', 2, 10)
get_timing('dev-clean', 2, 10)

stored_datasets = {}
get_timing('train-clean-100', 2, 10)
get_timing('train-clean-100', 2, 10)

# stored_datasets = {}
# get_timing('train-clean-360', 2, 10)
# get_timing('train-clean-360', 2, 10)
#
# stored_datasets = {}
# get_timing('train-other-500', 2, 10)

print('\n=============================================================================')
federated_dataset, user_ids = create_federated_dataset(
    split='dev-clean', trie=trie, stored_datasets=stored_datasets)
for i in range(3):
    user_dataset, seed = next(federated_dataset)
    print(f"\tReal user {user_dataset.user_id} has size of {len(user_dataset)}.")
    for idx, x in enumerate(user_dataset.iter(384000)):
        print(f"\t\tinput.shape: {x['input'].shape}   target.shape: {x['target'].shape}   total audio: {np.prod(x['input'].shape)}")
        if idx == 3:
            break
