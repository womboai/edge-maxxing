from random import randint
from threading import Thread, Event, Lock
from typing import Callable

from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights
from fiber.logging_utils import get_logger
from opentelemetry import trace
from substrateinterface import Keypair

from base.inputs_api import get_blacklist, get_inputs_state
from base.substrate_handler import SubstrateHandler
from base.system_info import SystemInfo
from weight_setting.contest_state import ContestState
from weight_setting.wandb_manager import WandbManager

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class WeightSetter:
    _thread: Thread
    _stop_flag: Event
    lock: Lock

    _epoch_length: int
    _substrate_handler: SubstrateHandler
    _metagraph: Metagraph
    _keypair: Keypair
    _uid: int
    _contest_state: Callable[[], ContestState]
    _api_hardware: list[SystemInfo]
    _wandb_manager: WandbManager
    _weights_version: int

    def __init__(
        self,
        version: str,
        epoch_length: int,
        substrate_handler: SubstrateHandler,
        metagraph: Metagraph,
        keypair: Keypair,
        uid: int,
        contest_state: Callable[[], ContestState],
        api_hardware: list[SystemInfo],
        wandb_manager: WandbManager,
    ):
        self._epoch_length = epoch_length
        self._substrate_handler = substrate_handler
        self._metagraph = metagraph
        self._keypair = keypair
        self._uid = uid
        self._contest_state = contest_state
        self._api_hardware = api_hardware
        self._wandb_manager = wandb_manager

        parts: list[str] = version.split(".")
        self._weights_version = int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])

        self._stop_flag = Event()
        self.lock = Lock()

        self._thread = Thread(target=self._run)
        self._thread.start()

    def shutdown(self):
        self._stop_flag.set()

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                with self.lock:
                    successful = self.set_weights()

                if successful:
                    logger.info(f"Successfully set weights, sleeping for {self._epoch_length} blocks")
                    self._stop_flag.wait(self._epoch_length * 12)
                else:
                    raise RuntimeError("Set weights attempt was unsuccessful")
            except Exception as e:
                blocks_to_sleep = randint(10, 50)
                logger.error(f"Failed to set weights, retrying in {blocks_to_sleep} blocks", exc_info=e)
                self._stop_flag.wait(blocks_to_sleep * 12)

    @tracer.start_as_current_span("set_weights")
    def set_weights(self) -> bool:
        contest_state = self._contest_state()
        inputs_state = get_inputs_state()

        if not contest_state:
            logger.warning("Will not set new weights as the contest state has not been set, setting to all ones")
            return self._set_equal_weights()

        delayed_weights = inputs_state.delayed_weight_setting
        benchmarks = contest_state.last_benchmarks if delayed_weights else contest_state.benchmarks

        if not delayed_weights and contest_state.get_untested_submissions():
            logger.info("Not setting new weights as benchmarking is not done, reusing old weights")
            delayed_weights = True
            benchmarks = contest_state.last_benchmarks

        if not contest_state.baseline:
            logger.warning("Will not set new weights as the baseline benchmarks have not been set, setting to all ones")
            return self._set_equal_weights()

        if not contest_state.last_benchmarks:
            if contest_state.benchmarks:
                logger.info("Setting weights to current benchmarks as the previous day's benchmarks have not been set")
                benchmarks = contest_state.benchmarks
            elif delayed_weights:
                logger.warning(
                    "Will not set new weights as the previous day's benchmarks have not been set, setting to all ones"
                    )
                return self._set_equal_weights()

        self._metagraph.sync_nodes()
        for hotkey, node in self._metagraph.nodes.items():
            if get_blacklist().is_blacklisted(hotkey, node.coldkey):
                contest_state.last_benchmarks.pop(hotkey, None)
                contest_state.submissions.pop(hotkey, None)
            if hotkey not in contest_state.submissions:
                contest_state.benchmarks.pop(hotkey, None)
                contest_state.last_benchmarks.pop(hotkey, None)

        scores = contest_state.get_scores(benchmarks)

        weights_by_key = contest_state.calculate_weights(scores)

        if not weights_by_key:
            logger.warning("Will not set new weights as all scores are equal, setting to all ones")

            return self._set_equal_weights()

        self._wandb_manager.send_metrics(contest_state, self._api_hardware)

        weights = [0.0] * len(self._metagraph.nodes)

        for key, weight in weights_by_key.items():
            node = self._metagraph.nodes.get(key)

            if node:
                weights[node.node_id] = weight

        return self._set_weights(weights)

    def _set_equal_weights(self) -> bool:
        return self._set_weights([1.0] * len(self._metagraph.nodes))

    def _set_weights(self, weights: list[float]) -> bool:
        return set_node_weights(
            self._substrate_handler.substrate,
            self._keypair,
            node_ids=list(range(len(weights))),
            node_weights=weights,
            netuid=self._metagraph.netuid,
            validator_node_id=self._uid,
            version_key=self._weights_version,
        )
