import argparse
import pandas as pd
import sox
from joblib import Parallel, delayed
import csv
import os
import mlx.data.datasets as datasets
from .common import get_latin_characters, process_latin_sentence
from unidecode import unidecode
from typing import List, Optional
import re
from collections import defaultdict
import tarfile


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


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)

    argument_parser.add_argument(
        '--cv_base_path',
        type=str,
        default=None,
        help='Base path to the uncompressed common-voice data.')

    argument_parser.add_argument(
        '--cv_archive_paths',
        nargs='+',
        default=[os.path.join(datasets.common.CACHE_DIR, 'cv-v13/cv-corpus-13.0-2023-03-09-en.tar.gz')],
        help='Paths to the tar or tar.gz archives with the common-voice data. '
             'If this flag is set, it is assumed that the archive was not yet '
             'uncompressed and the preprocessing will start by decompressing '
             'the archive. If flag is not set, it is assumed that the archive '
             'was uncompressed and the data can be accessed via csv_base_path.')

    argument_parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path where to save the tar archive with the processed data.')

    argument_parser.add_argument(
        '--languages',
        nargs='+',
        default=None,
        help='List of languages to process.')

    argument_parser.add_argument(
        '--num_threads',
        type=int,
        default=30,
        help="Number of threads. Set to number of cpus if 0.")

    arguments = argument_parser.parse_args()

    pwd = os.getcwd()

    if arguments.cv_base_path:
        assert arguments.cv_archive_paths is None or len(arguments.cv_archive_paths) == 0
        assert arguments.languages
        for language in arguments.languages:
            assert language in ['en', 'de', 'fr', 'es', 'pt']
        cv_base_path = arguments.cv_base_path
        languages = arguments.languages
    else:
        assert arguments.cv_archive_paths and len(arguments.cv_archive_paths) > 0
        assert arguments.languages is None or len(arguments.languages) == 0, 'Language will be automatically determined'
        languages = set()
        cv_base_path = None
        for cv_archive_path in arguments.cv_archive_paths:
            assert os.path.isfile(cv_archive_path)
            tar = None
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
                languages.add(language)
                # TODO: Maybe we can use tar directly (but could be slower)
                tar.extractall(path=os.path.dirname(cv_archive_path))
                tar.close()
            else:
                raise Exception('Unsupported archive type')
        languages = list(languages)
    #
    # cv_base_path = arguments.cv_base_path
    # if not arguments.cv_base_path:
    #     assert os.path.isfile(arguments.cv_archive_path)
    #     tar = None
    #     if arguments.cv_archive_path.endswith("tar.gz"):
    #         tar = tarfile.open(arguments.cv_archive_path, "r:gz")
    #     elif arguments.cv_archive_path.endswith("tar"):
    #         tar = tarfile.open(arguments.cv_archive_path, "r:")
    #     if tar:
    #         print(f'Unpacking {arguments.cv_archive_path}')
    #         cv_base_path = os.path.join(os.path.dirname(arguments.cv_archive_path), tar.next().name.split('/')[0])
    #         # TODO: Maybe we can use tar directly (but could be slower)
    #         tar.extractall(path=os.path.dirname(arguments.cv_archive_path))
    #         tar.close()
    #     else:
    #         raise Exception('Unsupported archive type')

    print('cv_base_path:', cv_base_path)
    preprocess_latin_text(cv_base_path, languages)
    process_audio_files(cv_base_path, languages, arguments.num_threads)

    def tar_filter_fn(x):
        if x.isdir():
            if x.name.endswith('clips'):
                return None
            return x
        elif x.name.endswith('.flac'):
            return x
        elif x.name.endswith('.tsv') and os.path.basename(x.name).startswith(f'{language}-'):
            return x

    if arguments.output_path:
        os.chdir(os.path.dirname(cv_base_path))
        data_path = cv_base_path.split('/')[-1]
        for language in languages:
            cv_lang_path = os.path.join(cv_base_path, language)
            output_tar_filename = f'{cv_base_path}-flac-{language}.tar'
            print(f'Compressing {output_tar_filename}')
            with tarfile.open(output_tar_filename, 'w') as tar:
                tar.add(os.path.join(data_path, language),
                        recursive=True,
                        filter=tar_filter_fn)

