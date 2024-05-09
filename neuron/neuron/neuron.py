from argparse import ArgumentParser
from os import makedirs
from os.path import expanduser, exists, join

from bittensor import metagraph, subtensor, config, wallet, logging

from pydantic import BaseModel

from loguru import logger


BASELINE_CHECKPOINT = "SimianLuo/LCM_Dreamshaper_v7"

AVERAGE_TIME = 30.0


class CheckpointInfo(BaseModel):
    repository: str
    average_time: float


class Neuron:
    config: config
    subtensor: subtensor
    metagraph: metagraph
    wallet: wallet

    def __init__(self, config: config):
        self.config = config

        r"""Checks/validates the config namespace object."""
        logging.check_config(config)

        full_path = expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                config.neuron.name,
            )
        )
        print("full path:", full_path)
        config.neuron.full_path = expanduser(full_path)
        if not exists(config.neuron.full_path):
            makedirs(config.neuron.full_path, exist_ok=True)

        if not config.neuron.dont_save_events:
            # Add custom event logger for the events.
            logger.level("EVENTS", no=38, icon="📝")
            logger.add(
                join(config.neuron.full_path, "events.log"),
                rotation=config.neuron.events_retention_size,
                serialize=True,
                enqueue=True,
                backtrace=False,
                diagnose=False,
                level="EVENTS",
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            )

        self.subtensor = subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = wallet(config=config)

        self.device = self.config.neuron.device

    @classmethod
    def add_args(cls, argument_parser: ArgumentParser):
        argument_parser.add_argument("--netuid", dest="netuid", type=int, required=False)

        argument_parser.add_argument(
            "--neuron.epoch_length",
            type=int,
            help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
            default=100,
        )

        argument_parser.add_argument(
            "--neuron.device",
            type=str,
            help="Device to run on.",
            default="mps",
        )

        argument_parser.add_argument(
            "--neuron.dont_save_events",
            action="store_true",
            help="If set, we dont save events to a log file.",
            default=False,
        )

        subtensor.add_args(argument_parser)
        wallet.add_args(argument_parser)

    def sync(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )

            exit()

        self.metagraph.sync(subtensor=self.subtensor)
