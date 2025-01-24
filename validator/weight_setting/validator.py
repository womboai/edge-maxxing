from datetime import timedelta
from importlib.metadata import version
from signal import signal, SIGINT, SIGHUP, SIGTERM
from threading import Event
from typing import Any

from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.logging_utils import get_logger
from opentelemetry import trace
from requests.exceptions import HTTPError, ConnectionError
from substrateinterface import SubstrateInterface, Keypair

from base.checkpoint import Uid, Submissions
from base.config import get_config
from base.contest import BenchmarkState
from base.inputs_api import get_inputs_state
from base.submissions import get_submissions
from base.substrate_handler import SubstrateHandler
from base.system_info import SystemInfo
from base_validator.api_data import BenchmarkingResults
from base_validator.auto_updater import AutoUpdater
from base_validator.telemetry import init_open_telemetry_logging
from weight_setting.wandb_manager import WandbManager
from .benchmarking_api import BenchmarkingApi, send_submissions_to_api
from .contest_state import ContestState
from .state_manager import StateManager
from .validator_args import add_args
from .weight_setter import WeightSetter

BENCHMARK_UPDATE_RATE_BLOCKS = 10

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class Validator:
    _stop_flag: Event = Event()
    auto_updater: AutoUpdater = AutoUpdater()
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

    substrate_handler = SubstrateHandler(substrate)

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

    wandb_manager: WandbManager

    weight_setter: WeightSetter
    benchmarking_apis: list[BenchmarkingApi]
    api_hardware: list[SystemInfo] = []

    def __init__(self):
        self.metagraph.sync_nodes()
        self.uid = list(self.metagraph.nodes.keys()).index(self.keypair.ss58_address)

        init_open_telemetry_logging({
            "service.name": "edge-maxxing-validator",
            "neuron.hotkey": self.keypair.ss58_address,
            "neuron.uid": self.uid,
            "neuron.signature": self.signature,
            "netuid": self.metagraph.netuid,
            "subtensor.chain_endpoint": self.substrate.url,
            "validator.version": self.validator_version,
        })

        self.wandb_manager = WandbManager(
            config=self.config,
            validator_version=self.validator_version,
            uid=self.uid,
            netuid=self.metagraph.netuid,
            hotkey=self.keypair.ss58_address,
            signature=self.signature,
        )

        contest_state = self.state_manager.load_state()
        if contest_state:
            self.contest_state = contest_state
            self.wandb_manager.init_wandb(self.contest_state)

        self.weight_setter: WeightSetter = WeightSetter(
            version=self.validator_version,
            epoch_length=self.config["epoch_length"],
            substrate_handler=self.substrate_handler,
            metagraph=self.metagraph,
            keypair=self.keypair,
            uid=self.uid,
            contest_state=lambda: self.contest_state,
            api_hardware=self.api_hardware,
            wandb_manager=self.wandb_manager,
        )

        self.benchmarking_apis = [BenchmarkingApi(api=api, keypair=self.keypair) for api in self.config["benchmarker_api"]]

        self.run()

    def initialize_apis(self, untested_submissions: Submissions):
        self.api_hardware.clear()
        for api in self.benchmarking_apis:
            api.initialize(
                uid=self.uid,
                signature=self.signature,
                netuid=self.metagraph.netuid,
                substrate_url=self.substrate.url,
            )
            self.api_hardware.append(api.hardware())
        send_submissions_to_api(
            version=self.validator_version,
            all_apis=self.benchmarking_apis,
            submissions=untested_submissions,
        )

    @tracer.start_as_current_span("initialize_contest")
    def initialize_contest(self, benchmarks_version: int):
        logger.info("Initializing contest")
        self.metagraph.sync_nodes()
        self.contest_state.start_new_contest(
            benchmarks_version=benchmarks_version,
            submissions=get_submissions(
                substrate_handler=self.substrate_handler,
                metagraph=self.metagraph,
            ),
        )

        if not self.contest_state.submissions:
            sleep_blocks = self.config["epoch_length"]
            logger.warning(f"No submissions found, sleeping for {sleep_blocks} blocks")
            self._stop_flag.wait(sleep_blocks * 12)
            return

        self.initialize_apis(self.contest_state.submissions)

        self.wandb_manager.init_wandb(self.contest_state)
        logger.info(f"Starting a new contest with {len(self.contest_state.submissions)} submissions")

    @tracer.start_as_current_span("do_step")
    def do_step(self):
        benchmarks_version = get_inputs_state().benchmarks_version

        if not self.contest_state:
            self.contest_state = ContestState.create(benchmarks_version)

        if self.contest_state.is_ended() or self.contest_state.benchmarks_version != benchmarks_version:
            with self.weight_setter.lock:
                self.initialize_contest(benchmarks_version)
            return

        untested_submissions = self.contest_state.get_untested_submissions()

        if not untested_submissions:
            self.contest_state.benchmarking_state = BenchmarkState.FINISHED
            self.state_manager.save_state(self.contest_state)
            self.wandb_manager.send_metrics(self.contest_state, self.api_hardware)
            self.contest_state.sleep_to_next_contest(self._stop_flag)
            return

        benchmarking_results = [api.results() for api in self.benchmarking_apis]

        if any(result.state == BenchmarkState.NOT_STARTED for result in benchmarking_results):
            self.initialize_apis(untested_submissions)
            return

        self.update_benchmarks(benchmarking_results)
        self.contest_state.benchmarking_state = BenchmarkState.IN_PROGRESS
        self._stop_flag.wait(BENCHMARK_UPDATE_RATE_BLOCKS * 12)

    def update_benchmarks(self, benchmarking_results: list[BenchmarkingResults]):
        if not self.contest_state:
            return

        baseline = benchmarking_results[0].baseline
        average_benchmarking_time = benchmarking_results[0].average_benchmarking_time

        if baseline and baseline != self.contest_state.baseline:
            logger.info(f"Updating baseline to {baseline}")
            self.contest_state.baseline = baseline

        for result in benchmarking_results:
            for key in result.benchmarks.keys() - self.contest_state.benchmarks.keys():
                logger.info(f"Updating benchmarks for {key}")
            for key in result.invalid_submissions - self.contest_state.invalid_submissions:
                logger.info(f"Marking submission from {key} as invalid")

            self.contest_state.benchmarks.update(result.benchmarks)
            self.contest_state.invalid_submissions.update(result.invalid_submissions)

        if average_benchmarking_time and average_benchmarking_time != self.contest_state.average_benchmarking_time:
            benchmarked = len(self.contest_state.benchmarks) + len(self.contest_state.invalid_submissions)
            eta = (len(self.contest_state.submissions) - benchmarked) * average_benchmarking_time
            logger.info(f"{benchmarked}/{len(self.contest_state.submissions)} benchmarked. Average benchmark time: {average_benchmarking_time:.2f}s, ETA: {timedelta(seconds=int(eta))}")
            self.contest_state.average_benchmarking_time = average_benchmarking_time

    def step(self):
        return self.contest_state.step if self.contest_state else 0

    def _shutdown(self, _signalnum, _handler):
        logger.info("Shutting down validator")
        self.weight_setter.shutdown()
        self.auto_updater.shutdown()
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
                if self.contest_state:
                    self.contest_state.step += 1
                    self.state_manager.save_state(self.contest_state)
                    self.wandb_manager.send_metrics(self.contest_state, self.api_hardware)
            except (ConnectionError, HTTPError) as e:
                logger.error(f"Error connecting to API, retrying in 10 blocks", exc_info=e)
                self._stop_flag.wait(BENCHMARK_UPDATE_RATE_BLOCKS * 12)
            except Exception as e:
                logger.error(f"Error during step {self.step()}", exc_info=e)

                self._stop_flag.wait(12)


def main():
    Validator()


if __name__ == '__main__':
    main()
