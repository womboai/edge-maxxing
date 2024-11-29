from random import randint
from threading import Thread, Event
from typing import Callable

from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights
from fiber.logging_utils import get_logger
from opentelemetry import trace
from substrateinterface import SubstrateInterface, Keypair

from base.inputs_api import blacklisted_keys, is_blacklisted
from weight_setting.contest_state import ContestState

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class WeightSetter:
    _thread: Thread
    _stop_flag: Event = Event()

    _epoch_length: int
    _substrate: Callable[[], SubstrateInterface]
    _metagraph: Metagraph
    _keypair: Keypair
    _uid: int
    _contest_state: Callable[[], ContestState]
    _weights_version: int

    def __init__(
        self,
        version: str,
        epoch_length: int,
        substrate: Callable[[], SubstrateInterface],
        metagraph: Metagraph,
        keypair: Keypair,
        uid: int,
        contest_state: Callable[[], ContestState],
    ):
        self._epoch_length = epoch_length
        self._substrate = substrate
        self._metagraph = metagraph
        self._keypair = keypair
        self._uid = uid
        self._contest_state = contest_state

        parts: list[str] = version.split(".")
        self._weights_version = int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])

        self._thread = Thread(target=self._run)
        self._thread.start()

    def shutdown(self):
        self._stop_flag.set()

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                if self.set_weights():
                    logger.info(f"Successfully set weights, sleeping for {self._epoch_length} blocks")
                    self._stop_flag.wait(self._epoch_length * 12)
                else:
                    raise RuntimeError("Set weights attempt was unsuccessful")
            except Exception as e:
                blocks_to_sleep = randint(2, 10)
                logger.error(f"Failed to set weights, retrying in {blocks_to_sleep} blocks: {e}")
                self._stop_flag.wait(blocks_to_sleep * 12)

    @tracer.start_as_current_span("set_weights")
    def set_weights(self) -> bool:
        contest_state = self._contest_state()

        if not contest_state:
            logger.error("Will not set new weights as the contest state has not been set, setting to all ones")
            return self._set_equal_weights()

        benchmarks = contest_state.last_benchmarks

        if not contest_state.baseline:
            logger.error("Will not set new weights as the baseline benchmarks have not been set, setting to all ones")
            return self._set_equal_weights()

        if not contest_state.last_benchmarks:
            if contest_state.benchmarks:
                logger.error("Setting weights to current benchmarks as the previous day's benchmarks have not been set")
                benchmarks = contest_state.benchmarks
            else:
                logger.error("Will not set new weights as the previous day's benchmarks have not been set, setting to all ones")
                return self._set_equal_weights()

        self._metagraph.sync_nodes()
        blacklist = blacklisted_keys()
        for hotkey, node in self._metagraph.nodes.items():
            if is_blacklisted(blacklist, hotkey, node.coldkey):
                benchmarks.pop(hotkey, None)
            if not hotkey in contest_state.submissions:
                benchmarks.pop(hotkey, None)

        scores = contest_state.get_scores()
        ranks = contest_state.get_ranks(scores)

        weights_by_key = contest_state.calculate_weights(ranks=ranks)

        return self._set_weights([
            weights_by_key.get(key, 0)
            for key in self._metagraph.nodes.keys()
        ])

    def _set_equal_weights(self) -> bool:
        return self._set_weights([1.0] * len(self._metagraph.nodes))

    def _set_weights(self, weights: list[float]) -> bool:
        return set_node_weights(
            self._substrate(),
            self._keypair,
            node_ids=list(range(len(self._metagraph.nodes))),
            node_weights=weights,
            netuid=self._metagraph.netuid,
            validator_node_id=self._uid,
            version_key=self._weights_version,
        )
