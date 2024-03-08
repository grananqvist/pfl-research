import argparse
import gzip
import os
import shutil
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

import mlx.data.datasets as datasets
import requests

# TODO: Add merged datasets (e.g. train-960(train-clean-100,train-clean-360,train-clean-500)).


def download_url(args):
    t0 = time.time()
    url, output_fname = args[0], args[1]
    try:
        r = requests.get(url, timeout=18000)  # timeout of 5 hours
        with open(output_fname, 'wb') as f:
            f.write(r.content)
        print(f'Downloaded {output_fname}')
        assert output_fname.endswith('.gz')
        print(f'Unzipping {output_fname}')
        with gzip.open(output_fname, 'rb') as f_in:
            with open(output_fname[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return (url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)


def download_parallel(urls, target_dir):
    cpus = cpu_count()
    print(f'{len(urls)} urls to download')
    target_fnames = [
        os.path.join(target_dir, os.path.basename(urlparse(url).path))
        for url in urls
    ]
    args = zip(urls, target_fnames)
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)
    for result in results:
        print(f'Processed {result[0]} in {result[1]} seconds')


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--splits',
        nargs='+',
        default=[
            "dev-clean", "dev-other", "test-clean", "test-other",
            "train-clean-100"
        ],
        help='List of data splits to download, e.g. "dev-clean dev-other".')
    argument_parser.add_argument(
        '--num_threads',
        type=int,
        default=0,
        help="Number of threads. Set to number of cpus if 0.")

    args = argument_parser.parse_args()

    allowable_splits = list(datasets.librispeech.SPLITS.keys())
    for split in args.splits:
        assert split in datasets.librispeech.SPLITS, f"Allowable splits are {allowable_splits}"

    urls = [datasets.librispeech.SPLITS[split][0] for split in args.splits]

    target_dir = os.path.join(datasets.common.CACHE_DIR, "librispeech")

    os.makedirs(target_dir, exist_ok=True)

    download_parallel(urls, target_dir)
