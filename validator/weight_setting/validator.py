from importlib.metadata import version
from signal import signal, SIGINT, SIGHUP, SIGTERM
from threading import Event
from time import sleep
from typing import Any

from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.logging_utils import get_logger
from opentelemetry import trace
from requests.exceptions import HTTPError, ConnectionError
from substrateinterface import SubstrateInterface, Keypair

from base.checkpoint import Uid
from base.config import get_config
from base.submissions import get_submissions
from base_validator.api_data import BenchmarkState, BenchmarkingResults
from base_validator.auto_updater import AutoUpdater
from base_validator.telemetry import init_open_telemetry_logging
from weight_setting.wandb_manager import WandbManager
from .benchmarking_api import BenchmarkingApi, send_submissions_to_api
from .contest_state import ContestState
from .state_manager import StateManager
from .validator_args import add_args
from .weight_setter import WeightSetter

BENCHMARK_UPDATE_RATE_BLOCKS = 10
BENCHMARKS_VERSION = 1

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

class Validator:
    _stop_flag: Event = Event()
    contest_state: ContestState | None = None
    validator_version: str = version("edge-maxxing-validator")
    uid: Uid

    config: dict[str, Any] = get_config(add_args)

    keypair: Keypair = load_hotkey_keypair(
        wallet_name=config["wallet.name"],
        hotkey_name=config["wallet.hotkey"],
    )

    signature = f"0x{keypair.sign(f'{validator_version}:{keypair.ss58_address}').hex()}"

    substrate: SubstrateInterface = get_substrate(
        subtensor_network=config["subtensor.network"],
        subtensor_address=config["subtensor.chain_endpoint"]
    )

    metagraph: Metagraph = Metagraph(
        substrate=substrate,
        netuid=config["netuid"],
        load_old_nodes=False,
    )

    state_manager: StateManager = StateManager(
        wallet_name=config["wallet.name"],
        hotkey_name=config["wallet.hotkey"],
        netuid=metagraph.netuid,
    )

    wandb_manager: WandbManager = WandbManager(
        config=config,
        validator_version=validator_version,
        uid=metagraph.netuid,
        netuid=metagraph.netuid,
        hotkey=keypair.ss58_address,
        signature=signature,
    )

    weight_setter: WeightSetter
    benchmarking_apis: list[BenchmarkingApi]

    def __init__(self):
        self.metagraph.sync_nodes()
        self.uid = list(self.metagraph.nodes.keys()).index(self.keypair.ss58_address)

        self.weight_setter: WeightSetter = WeightSetter(
            version=self.validator_version,
            epoch_length=self.config["epoch_length"],
            substrate=lambda: self.substrate,
            metagraph=self.metagraph,
            keypair=self.keypair,
            uid=self.uid,
            contest_state=lambda: self.contest_state,
        )

        self.contest_state = self.state_manager.load_state()

        self.benchmarking_apis = [BenchmarkingApi(api=api, keypair=self.keypair) for api in self.config["benchmarker_api"]]

        init_open_telemetry_logging({
            "neuron.uid": self.uid,
            "neuron.signature": self.signature,
            "subtensor.chain_endpoint": self.substrate.url,
            "validator.version": self.validator_version,
        })

        self.run()

    @tracer.start_as_current_span("initialize_contest")
    def initialize_contest(self):
        for api in self.benchmarking_apis:
            api.initialize(
                uid=self.uid,
                signature=self.signature,
                substrate_url=self.substrate.url,
            )

        self.metagraph.sync_nodes()
        self.contest_state.start_new_contest(
            benchmarks_version=BENCHMARKS_VERSION,
            submissions=get_submissions(
                substrate=self.substrate,
                metagraph=self.metagraph,
                block=self.substrate.get_block_number(None),  # type: ignore
            )
        )
        self.wandb_manager.init_wandb(self.contest_state)

    @tracer.start_as_current_span("do_step")
    def do_step(self):
        if not self.contest_state:
            self.contest_state = ContestState.create(BENCHMARKS_VERSION)

        if self.contest_state.is_ended() or self.contest_state.benchmarks_version != BENCHMARKS_VERSION:
            self.initialize_contest()
            return

        untested_submissions = self.contest_state.get_untested_submissions()

        if not untested_submissions:
            self.contest_state.sleep_to_next_contest(self._stop_flag)
            return

        benchmarking_results = [api.results() for api in self.benchmarking_apis]

        if any(result.state == BenchmarkState.NOT_STARTED for result in benchmarking_results):
            send_submissions_to_api(
                version=self.validator_version,
                all_apis=self.benchmarking_apis,
                submissions=untested_submissions,
            )
            return

        self.update_benchmarks(benchmarking_results)
        sleep(BENCHMARK_UPDATE_RATE_BLOCKS * 12)

    def update_benchmarks(self, benchmarking_results: list[BenchmarkingResults]):
        self.contest_state.baseline = benchmarking_results[0].baseline
        self.contest_state.average_benchmarking_time = benchmarking_results[0].average_benchmarking_time

        for result in benchmarking_results:
            self.contest_state.benchmarks.update(result.benchmarks)
            self.contest_state.invalid_submissions.update(result.invalid_submissions)

    def step(self):
        return self.contest_state.step if self.contest_state else 0

    def _shutdown(self, _signalnum, _handler):
        logger.info("Shutting down validator")
        self.weight_setter.stop()
        self._stop_flag.set()

    def run(self):
        logger.info("Initializing validator")
        signal(SIGTERM, self._shutdown)
        signal(SIGINT, self._shutdown)
        signal(SIGHUP, self._shutdown)
        while not self._stop_flag.is_set():
            try:
                logger.info(f"Step {self.step()}")
                self.do_step()
                self.contest_state.step += 1
                self.state_manager.save_state(self.contest_state)
                self.wandb_manager.send_metrics(self.contest_state)
            except (ConnectionError, HTTPError) as e:
                logger.error(f"Error connecting to API, retrying in 10 blocks: {e}")
                self._stop_flag.wait(BENCHMARK_UPDATE_RATE_BLOCKS * 12)
            except Exception as e:
                logger.error(f"Error during step {self.step()}", exc_info=e)
                self.substrate = get_substrate(subtensor_address=self.substrate.url)

def main():
    AutoUpdater()
    Validator()

if __name__ == '__main__':
    main()
