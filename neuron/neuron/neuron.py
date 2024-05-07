from argparse import ArgumentParser

from bittensor import metagraph, subtensor, config, wallet


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
        metagraph.add_args(argument_parser)
        wallet.add_args(argument_parser)
