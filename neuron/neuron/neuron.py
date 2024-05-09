from argparse import ArgumentParser

from bittensor import metagraph, subtensor, config, wallet, logging

from pydantic import BaseModel


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
