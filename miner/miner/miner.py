from argparse import ArgumentParser

from bittensor import axon, config, logging
from bittensor.utils.networking import get_external_ip

from neuron.neuron.neuron import Neuron
from neuron.neuron.pipeline import load_pipeline


class Miner(Neuron):
    def __init__(self, config: config):
        super().__init__(config)

        self.pipeline, self.checkpoint_path = load_pipeline(self.device)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )

        if self.config.blacklist.allow_non_registered:
            logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        external_ip = self.config.axon.external_ip or get_external_ip()
        external_port = self.config.axon.external_port or self.config.axon.port

        self.subtensor.serve(
            wallet=self.wallet,
            ip=external_ip,
            port=external_port,
            protocol=4,
            netuid=config.netuid,
        )

    @classmethod
    def add_args(cls, argument_parser: ArgumentParser):
        super().add_args(argument_parser)

        axon.add_args(argument_parser)

        argument_parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        argument_parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        argument_parser.add_argument(
            "--blacklist.validator_minimum_tao",
            type=int,
            help="The minimum number of TAO needed for a validator's queries to be accepted.",
            default=4096,
        )
