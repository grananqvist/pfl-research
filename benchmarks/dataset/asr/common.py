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


class ASRDataset:

    def __init__(
        self,
        dataset: str,
        data_path: str,
        split: str,
        trie: dx.core.CharTrie,
        target_pad: bool = False,
        n_threads: int = 4,
        dynamic_batching: bool = True,
        stored_datasets: Optional[Dict] = None,
    ):
        self.trie = trie
        if dataset == 'librispeech':
            self.initialize_librispeech(
                name=os.path.join(data_path, f'{split}.tar'),
                prefix=f'LibriSpeech/{split}',
                trie=trie,
                stored_datasets=stored_datasets,
                n_threads=n_threads,
                target_pad=target_pad,
                dynamic_batching_key='input_length'
                if dynamic_batching else None,
            )
        else:
            raise ValueError(f'Unknown dataset {dataset}')

    def initialize_librispeech(
            self,
            name: str,
            prefix: str,
            trie: dx.core.CharTrie,
            stored_datasets,  # TODO: Do we only need this for the central dataset?
            n_threads=-1,
            target_pad=False,
            dynamic_batching_key=None):
        assert trie is not None

        self._dynamic_batching_key = dynamic_batching_key

        # update stored_datasets
        if stored_datasets is not None and name in stored_datasets:
            self.dataset = stored_datasets[name]
        else:
            start = time.time()
            self.dataset = (
                dx.files_from_tar(name).to_stream().sample_transform(
                    lambda s: s if bytes(s["file"]).endswith(b".txt") else {}).
                read_from_tar(
                    name,
                    "file",
                    "samples",
                ).line_reader_from_key(
                    "samples", "sample", from_memory=True).sample_transform(
                        self.process_csv_row).prefetch(
                            n_threads,
                            n_threads)  # TODO: check where to prefetch
                .read_from_tar(name, "audio_file", "audio",
                               prefix=prefix).load_audio(
                                   "audio",
                                   from_memory=True,
                                   output_key="input").tokenize(
                                       "transcript",
                                       trie,
                                       ignore_unk=True,
                                       output_key="target").shape(
                                           "input", "input_length",
                                           0).squeeze("input",
                                                      -1)  # %1s, one channel
            )
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
            *([item['input_length'].item(), item['user_id'].item()]
              for item in self.dataset))

        end = time.time()
        logger.info(
            f'Time for extracting durations and client ids: {end - start}')

        # duration_perm = np.argsort(np.array(durations))[rank::num_nodes]
        # durations = np.array(durations)[duration_perm]
        # client_ids = np.array(client_ids)[duration_perm]
        # dataset = dataset.perm(duration_perm)
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

        return {
            "audio_file": audio_path,
            "transcript": transcript,
            "user_id": user_id
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
        return MlxDataUserDataset(
            dataset,
            user_id=user_id,
            dynamic_batching_key=self._dynamic_batching_key,
            trie=self.trie)

    def full_dataset(self):
        dataset = self.dataset
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
        if dynamic_batching_key:
            self._dynamic_batching_values = np.array(
                [item[dynamic_batching_key].item() for item in raw_data])
        self._user_id = user_id
        self._metadata = metadata or {}
        self._train_kwargs = train_kwargs or {}
        self._eval_kwargs = eval_kwargs or {}
        self._shuffle = shuffle

    def __len__(self):
        """
        Returns the size of the raw data stored in the buffer.
        """
        return self._raw_data.size()

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        if batch_size is None:
            # No batch size, so consider entire dataset as a single batch.
            batch_size = len(self)
            dataset = self._raw_data.batch(batch_size=batch_size)
            yield dataset

        dataset = self._raw_data
        if self._shuffle:
            dataset = dataset.shuffle()
        dataset = dataset.to_stream()
        if self._dynamic_batching_key:
            dataset = dataset.dynamic_batch(
                buffer_size=500,  # stream_buffer_size
                key=self._dynamic_batching_key,
                max_data_size=batch_size,
                shuffle=False,
                pad={
                    "input": 0,
                    "target": self.trie.search("@").id
                },
            )
        else:
            dataset = dataset.batch(batch_size,
                                    pad={
                                        "input": 0,
                                        "target": self.trie.search("@").id
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

        raw_data_as_array = np.array(self._raw_data)

        left_raw_data = raw_data_as_array[left_slice]
        right_raw_data = raw_data_as_array[right_slice]
        train_dataset = MlxDataUserDataset(
            dx.buffer_from_vector(left_raw_data),
            trie=self.trie,
            user_id=self.user_id,
            metadata=self.metadata,
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
        val_dataset = MlxDataUserDataset(
            dx.buffer_from_vector(right_raw_data),
            trie=self.trie,
            user_id=self.user_id,
            metadata=self.metadata,
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
        return train_dataset, val_dataset

    def get_worker_partition(self) -> 'Dataset':
        partition_range = get_ops().distributed.distribute_range(len(self))
        return MlxDataUserDataset(
            trie=self.trie,
            raw_data=dx.buffer_from_vector(
                np.array(self._raw_data)[partition_range]),
            train_kwargs=self.train_kwargs,
            eval_kwargs=self.eval_kwargs,
            shuffle=self._shuffle,
            dynamic_batching_key=self._dynamic_batching_key)
