import mlx.data as dx


splits = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--splits',
        help=('Output directory for the original files '
              'and the processed pickle files.'),
        default='./data/cifar10')
    arguments = argument_parser.parse_args()

    dl_preprocess_and_dump(arguments.output_dir)
