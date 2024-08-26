from argparse import ArgumentParser

import bittensor as bt
from huggingface_hub import snapshot_download


def main():
    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "--repository",
        type=str,
        help="The repository to download the files of",
    )

    config = bt.config(argument_parser)

    snapshot_download(config.repository, cache_dir="../inference/models")


if __name__ == '__main__':
    main()
