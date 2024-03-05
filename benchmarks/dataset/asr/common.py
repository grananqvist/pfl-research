import logging
import os.path
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.data as dx
import numpy as np
import pandas as pd

from pfl.data.dataset import Dataset
from pfl.internal.ops.selector import get_framework_module as get_ops
from pfl.model.pytorch import PyTorchModel  # pylint: disable=unused-import

logger = logging.getLogger(name=__name__)

USE_MLX_DYNAMIC_BATCHING = False


class ASRDataset:

    def __init__(  # TODO: Remove defaults as appropriate
        self,
        dataset: str,
        data_path: str,
        split: str,
        trie: dx.core.CharTrie,
        target_pad: bool,
        n_threads: int,
        dynamic_batching: bool = True,
        stored_datasets: Optional[Dict] = None,
        max_sample_audio_length: Optional[int] = None,
        lazy_load_audio: bool = False,
    ):
        self.trie = trie
        self.lazy_load_audio = lazy_load_audio
        if dataset == 'librispeech':
            self.initialize_librispeech(
                name=os.path.join(data_path, f'{split}.tar'),
                prefix=f'LibriSpeech/{split}',
                trie=trie,
                stored_datasets=stored_datasets,
                n_threads=n_threads,
                target_pad=target_pad,
                dynamic_batching_key=('input' if dynamic_batching else None),
                max_sample_audio_length=max_sample_audio_length,
            )
        elif dataset == 'cv-en-v13':
            self.initialize_common_voice(
                name=os.path.join(data_path,
                                  'cv-corpus-13.0-2023-03-09-en.tar'),
                prefix='cv-corpus-13.0-2023-03-09/en',
                split=split,
                trie=trie,
                stored_datasets=stored_datasets,
                n_threads=n_threads,
                target_pad=target_pad,
                dynamic_batching_key=('input' if dynamic_batching else None),
                max_sample_audio_length=max_sample_audio_length,
            )
        else:
            raise ValueError(f'Unknown dataset {dataset}')

    def initialize_librispeech(  # TODO: Remove defaults
        self,
        name: str,
        prefix: str,
        trie: dx.core.CharTrie,
        # TODO: Unless we have to access same dataset repeatedly, we won't need cache.
        #   Check final solution.
        stored_datasets,
        n_threads=-1,
        target_pad=False,
        dynamic_batching_key=None,
        max_sample_audio_length: Optional[int] = None, # TODO: Make this work.
    ):
        assert trie is not None

        self._dynamic_batching_key = dynamic_batching_key

        # update stored_datasets
        if stored_datasets is not None and name in stored_datasets:
            self.dataset = stored_datasets[name]
        else:
            start = time.time()
            self.dataset = dx.files_from_tar(name).to_stream()
            self.dataset = self.dataset.sample_transform(
                lambda s: s if bytes(s["file"]).endswith(b".txt") else {})
            self.dataset = self.dataset.read_from_tar(name, "file", "samples")
            self.dataset = self.dataset.line_reader_from_key("samples", "sample", from_memory=True)
            self.dataset = self.dataset.sample_transform(self.librispeech_process_csv_row)
            self.dataset = self.dataset.prefetch(n_threads, n_threads)  # TODO: check where to prefetch
            self.dataset = self.dataset.read_from_tar(name, "audio_file", "audio", prefix=prefix)
            self.dataset = self.dataset.load_audio("audio", from_memory=True, output_key="input")
            self.dataset = self.dataset.filter_key("audio", remove=True)
            self.dataset = self.dataset.filter_key("audio_file", remove=True)
            self.dataset = self.dataset.tokenize("transcript", trie, ignore_unk=True, output_key="target")
            self.dataset = self.dataset.shape("input", "input_length", 0)
            self.dataset = self.dataset.squeeze("input", -1)  # %1s, one channel

            # self.dataset = (
            #     dx.files_from_tar(name).to_stream().sample_transform(
            #         lambda s: s if bytes(s["file"]).endswith(b".txt") else {}).
            #     read_from_tar(
            #         name,
            #         "file",
            #         "samples",
            #     ).line_reader_from_key(
            #         "samples", "sample", from_memory=True).sample_transform(
            #             self.librispeech_process_csv_row).prefetch(
            #                 n_threads,
            #                 n_threads)  # TODO: check where to prefetch
            #     .read_from_tar(name, "audio_file", "audio",
            #                    prefix=prefix).load_audio(
            #                        "audio",
            #                        from_memory=True,
            #                        output_key="input")
            #     .filter_key("audio", remove=True)
            #     .filter_key("audio_file", remove=True)
            #     .tokenize(
            #                            "transcript",
            #                            trie,
            #                            ignore_unk=True,
            #                            output_key="target").shape(
            #                                "input", "input_length",
            #                                0).squeeze("input",
            #                                           -1)  # %1s, one channel
            # )
            if max_sample_audio_length is not None:
                self.dataset = self.dataset.sample_transform(
                    lambda sample: sample if sample['input_length'].item(
                    ) <= max_sample_audio_length else {})
            if target_pad:
                self.dataset = self.dataset.pad('target', 0, 1, 1,
                                                1)  # pad target with silence

            self.dataset = self.dataset.shape('target', 'target_length', 0)

            # prefetch all data and then bufferize them
            self.dataset = self.dataset.prefetch(n_threads,
                                                 n_threads).to_buffer()

            end = time.time()
            logger.info(
                f'Time for initializing the dataset buffer: {end - start}')

            if stored_datasets is not None:
                stored_datasets[name] = self.dataset

        start = time.time()

        self.durations, self.client_ids = zip(
            *([item['input_length'].item(),
               bytes(item['user_id'])] for item in self.dataset
              if 'user_id' in item))

        print('max audio length:', np.max(self.durations))

        end = time.time()
        logger.info(
            f'Time for extracting durations and client ids: {end - start}')

        self.durations = np.array(self.durations)
        self.client_ids = np.array(self.client_ids)

        # TODO: Check why not exact match for total duration e.g. for train-clean-100
        logger.info(
            f'Dataset total (h) duration {np.sum(self.durations) / 16000 / 60 / 60}'
        )

        start = time.time()

        self.clients_unique = np.unique(self.client_ids)
        logger.info(f'Total {len(self.clients_unique)} clients')

        self.df = pd.DataFrame({
            "duration": self.durations,
            "client": self.client_ids
        })
        self.df_group = self.df.groupby("client", sort=False).groups

        end = time.time()
        logger.info(
            f'Time for grouping and other postprocessing: {end - start}')

    def initialize_common_voice(  # TODO: Remove defaults
            self,
            name: str,
            prefix: str,
            split: str,
            trie: dx.core.CharTrie,
            stored_datasets,  # TODO: Do we only need this for the central dataset?
            n_threads: int = -1,
            target_pad: bool = False,
            dynamic_batching_key: Optional[str] = None,
            max_sample_audio_length: Optional[int] = None):
        assert trie is not None

        self._dynamic_batching_key = dynamic_batching_key

        # update stored_datasets
        if stored_datasets is not None and name in stored_datasets:
            self.dataset = stored_datasets[name]
        else:
            start = time.time()

            self.dataset = dx.buffer_from_vector([{'file': bytes(f'{prefix}/{split}.tsv', 'utf-8')}])
            self.dataset = self.dataset.read_from_tar(name, 'file', 'samples').to_stream()
            self.dataset = self.dataset.line_reader_from_key('samples', 'sample', from_memory=True)
            self.dataset = self.dataset.prefetch(n_threads, n_threads)
            self.dataset = self.dataset.sample_transform(self.common_voice_process_csv_row)
            self.dataset = self.dataset.prefetch(n_threads, n_threads).read_from_tar(
                                name,
                                "audio_file",
                                "audio",
                                prefix=f'{prefix}/clips')
            self.dataset = self.dataset.prefetch(n_threads, n_threads).load_audio(
                                    "audio",
                                    from_memory=True,
                                    output_key="input",
                                    sample_rate=16000)
            self.dataset = self.dataset.filter_key("audio", remove=True)
            self.dataset = self.dataset.filter_key("audio_file", remove=True)
            self.dataset = self.dataset.tokenize(
                                        "transcript",
                                        trie,
                                        ignore_unk=True,
                                        output_key="target")
            self.dataset = self.dataset.shape("input", "input_length", 0)
            self.dataset = self.dataset.squeeze("input", -1)


            # self.dataset = (dx.buffer_from_vector([{
            #     'file':
            #     bytes(f'{prefix}/{split}.tsv', 'utf-8')
            # }]).read_from_tar(
            #     name, 'file', 'samples').to_stream().line_reader_from_key(
            #         'samples', 'sample', from_memory=True).prefetch(
            #             n_threads, n_threads).sample_transform(
            #                 self.common_voice_process_csv_row).prefetch(n_threads, n_threads).read_from_tar(
            #                     name,
            #                     "audio_file",
            #                     "audio",
            #                     prefix=f'{prefix}/clips').prefetch(n_threads, n_threads).load_audio(
            #                         "audio",
            #                         from_memory=True,
            #                         output_key="input",
            #                         sample_rate=16000)
            #                 .filter_key("audio", remove=True)
            #                 .filter_key("audio_file", remove=True)
            #                 .tokenize(
            #                             "transcript",
            #                             trie,
            #                             ignore_unk=True,
            #                             output_key="target").shape(
            #                                 "input", "input_length",
            #                                 0).squeeze("input", -1))
            if max_sample_audio_length is not None:
                self.dataset = self.dataset.sample_transform(
                    lambda sample: sample if sample['input_length'].item(
                    ) <= max_sample_audio_length else {})
            if target_pad:
                self.dataset = self.dataset.pad('target', 0, 1, 1,
                                                1)  # pad target with silence

            self.dataset = self.dataset.shape('target', 'target_length', 0)

            # prefetch all data and then bufferize them
            self.dataset = self.dataset.prefetch(n_threads,
                                                 n_threads).to_buffer()

            end = time.time()
            logger.info(
                f'Time for initializing the dataset buffer: {end - start}')

            # total_audio = 0.0
            # total_input = 0.0
            # for item in self.dataset:
            #     total_audio += item['audio'].nbytes / 1024.0
            #     total_input += item['input'].nbytes / 1024.0
            # print(f'ratio audio/input: {total_audio/total_input} = {total_audio} / {total_input}')

            print(
                'max audio length:',
                np.max([
                    sample['input_length'].item() for sample in self.dataset
                ]))


            if stored_datasets is not None:
                stored_datasets[name] = self.dataset

        print('self.dataset:', self.dataset)
        print('First 3 items of self.dataset')
        for index in range(3):
            print(
                f"self.dataset[index]['user_id']: {bytes(self.dataset[index]['user_id'])}"
            )
            print(
                f"self.dataset[index]['input_length']: {self.dataset[index]['input_length']}   type: {type(self.dataset[index]['input_length'])}"
            )

        start = time.time()

        self.durations, self.client_ids = zip(
            *([item['input_length'].item(),
               bytes(item['user_id'])] for item in self.dataset))

        end = time.time()
        logger.info(
            f'Time for extracting durations and client ids: {end - start}')

        self.durations = np.array(self.durations)
        self.client_ids = np.array(self.client_ids)

        # TODO: Check why not exact match for total duration e.g. for train-clean-100
        logger.info(
            f'Dataset total (h) duration {np.sum(self.durations) / 16000 / 60 / 60}'
        )

        start = time.time()

        self.clients_unique = np.unique(self.client_ids)
        logger.info(f'Total {len(self.clients_unique)} clients')

        self.df = pd.DataFrame({
            "duration": self.durations,
            "client": self.client_ids
        })
        self.df_group = self.df.groupby("client", sort=False).groups

        end = time.time()
        logger.info(
            f'Time for grouping and other postprocessing: {end - start}')

    @staticmethod
    def librispeech_process_csv_row(sample):
        # Split the line
        file_part, transcript = bytes(sample["sample"]).split(b" ", 1)

        # Extract the audio path
        parts = file_part.split(b"-")
        parts[-1] = file_part + b".flac"
        audio_path = b"/".join(parts)

        # Prepare the transcript
        transcript = transcript.lower()

        # User id
        user_id = parts[-3]

        return {
            "audio_file": audio_path,
            "transcript": transcript,
            "user_id": user_id
        }

    @staticmethod
    def common_voice_process_csv_row(sample):
        # Split the line
        str_list = bytes(sample['sample']).split(b'\t')
        # print('str_list:', str_list)
        if str_list[0] == b'client_id':
            assert str_list[1] == b'path'
            assert str_list[2] == b'sentence'
            return {}  # skip the header
        return {
            'user_id': str_list[0],
            'audio_file': str_list[1],
            'transcript':
            str_list[2].lower(),  # TODO: More preprocessing here?
        }

    def get_user_ids(self):
        return self.clients_unique

    def get_user_dataset(self, user_id):
        # We will shuffle inside dataset iter before serving batches so do not have to here
        tmp = self.df_group[user_id]
        client_permutation = tmp

        dataset = self.dataset.perm(client_permutation)

        return dataset

    def make_dataset_fn(self, user_id):
        dataset = self.get_user_dataset(user_id)
        # TODO: Should also get rid of unneeded fields 'audio', 'audio_file', and 'user_id'.
        # dataset = dataset.filter_key('audio', remove=True) #.filter_key('user_id')
        # print(f'make_dataset_fn: size={dataset.size()}  self._dynamic_batching_key={self._dynamic_batching_key}')
        # print(f'local dataset after filtering: {dataset}')
        return MlxDataUserDataset(
            dataset,
            user_id=user_id,
            dynamic_batching_key=self._dynamic_batching_key,
            trie=self.trie)

    def full_dataset(self):
        dataset = self.dataset
        # dataset = dataset.filter_key('audio', remove=True) #.filter_key('user_id')
        # print(f'full dataset after filtering: {dataset}')
        # TODO: Should also get rid of unneeded fields 'audio', 'audio_file', and 'user_id'.
        return MlxDataUserDataset(
            dataset,
            dynamic_batching_key=self._dynamic_batching_key,
            trie=self.trie)


class MlxDataUserDataset(Dataset):
    """
    A representation of a user dataset based on mlx.data that allows dynamic batching.

    :param raw_data:
        An mlx-data buffer with preprocessed raw data (these are not padded or
        batched).
    :param user_id:
        (Optional) String user identifier.
    :param metadata:
        (Optional) Store additional data about the user. Can be retrieved
        later by the algorithm.
    :param train_kwargs:
        (Optional) A dictionary of any additional training parameters to unpack
        in the training call of the deep learning framework.
    :param eval_kwargs:
        (Optional) A dictionary of any additional evaluation parameters to unpack
        in the evaluation call of the deep learning framework.
    :param shuffle:
        Shuffle the data before producing batches.
    :param dynamic_batching_key:
        When specified, dynamic batching will be used based on the scalars provided
        in this key.
    """

    def __init__(self,
                 raw_data: dx.Buffer,
                 trie: dx.core.CharTrie,
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 train_kwargs: Optional[Dict[str, Any]] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None,
                 shuffle: bool = True,
                 dynamic_batching_key: Optional[str] = None):
        self._raw_data = raw_data
        self._dynamic_batching_key = dynamic_batching_key
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._shuffle = shuffle
        self._trie = trie

    def __len__(self):
        """
        Returns the size of the raw data stored in the buffer.
        """
        return self._raw_data.size()

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        # logger.info(f'Batching iterator got batch_size={batch_size}')
        if batch_size is None:
            # No batch size, so consider entire dataset as a single batch.
            batch_size = len(self)
            dataset = self._raw_data.batch(batch_size=batch_size)
            yield dataset

        def get_dynamic_batch_sizes(batch_size, dynamic_batching_values):
            batch_sizes = []
            current_max_duration = 0
            current_batch_size = 0
            for this_value in dynamic_batching_values:
                new_max = max(current_max_duration, this_value)
                new_batch = current_batch_size + 1
                # TODO: if the duration of one sample audio is longer than
                #  the batch size, we create a batch with a single member,
                #  possibly leading to issues.
                if new_batch * new_max > batch_size and current_batch_size > 0:
                    batch_sizes.append(current_batch_size)
                    # print('new batch duration:', current_max_duration * current_batch_size,
                    #       '   batch size:', current_batch_size)
                    current_batch_size = 0
                    current_max_duration = 0
                current_batch_size += 1
                current_max_duration = max(current_max_duration, this_value)
            if current_batch_size > 0:
                batch_sizes.append(current_batch_size)

            return batch_sizes

        dataset = self._raw_data
        # if self._shuffle:
        #     dataset = dataset.shuffle()

        # print('dataset max input length:', np.max([x['input_length'].item() for x in dataset]))
        # print('users:', set([x['user_id'].item() for x in dataset]))
        #

        if self._dynamic_batching_key:
            if USE_MLX_DYNAMIC_BATCHING:
                # print(f'dynamic_batch uses self._dynamic_batching_key={self._dynamic_batching_key}, batch_size={batch_size}')
                dataset = dataset.to_stream().dynamic_batch(
                    buffer_size=8,  # stream buffer_size
                    key=self._dynamic_batching_key,
                    max_data_size=batch_size,
                    shuffle=True,
                    pad={
                        "input": 0,
                        "target": self._trie.search("@").id
                    },
                    num_threads=2,
                )
            else:
                dynamic_batching_values = [
                    x[self._dynamic_batching_key].shape[0]
                    for x in self._raw_data
                ]
                # print('dynamic_batching_values:', dynamic_batching_values)
                batch_sizes = get_dynamic_batch_sizes(batch_size,
                                                      dynamic_batching_values)
                dataset = dataset.batch(batch_sizes)
        else:
            if self._shuffle:
                dataset = dataset.shuffle()
            dataset = dataset.batch(batch_size,
                                    pad={
                                        "input": 0,
                                        "target": self._trie.search("@").id
                                    })
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

        left_raw_data = self._raw_data.perm(left_slice)
        right_raw_data = self._raw_data.perm(right_slice)
        train_dataset = MlxDataUserDataset(
            dx.buffer_from_vector(left_raw_data),
            trie=self._trie,
            user_id=self.user_id,
            metadata=self.metadata,
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
        val_dataset = MlxDataUserDataset(
            dx.buffer_from_vector(right_raw_data),
            trie=self._trie,
            user_id=self.user_id,
            metadata=self.metadata,
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
        return train_dataset, val_dataset

    def get_worker_partition(self) -> 'Dataset':
        #        print('self._raw_data:', self._raw_data)
        partition_range = get_ops().distributed.distribute_range(len(self))
        print('get_worker_partition partition_range:', partition_range,
              'total:', len(self))
        return MlxDataUserDataset(
            trie=self._trie,
            raw_data=self._raw_data.perm(partition_range),
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
