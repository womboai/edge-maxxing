import json
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date, datetime
from operator import itemgetter, attrgetter
from os import makedirs
from os.path import isfile
from pathlib import Path
from threading import Thread
from typing import cast, TypeAlias, Any
from zoneinfo import ZoneInfo

import numpy
import requests
import wandb
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights
from fiber.logging_utils import get_logger
from numpy import real, isreal
from numpy.polynomial import Polynomial
from pickle import dump, load

from pydantic import RootModel
from substrateinterface import SubstrateInterface, Keypair
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
from websockets import ConnectionClosedError
from websockets.sync.client import connect, ClientConnection

from neuron import (
    CheckpointSubmission,
    get_config,
    ContestId,
    CURRENT_CONTEST,
    find_contest,
    ContestDeviceValidationError,
    Contest,
    Key,
    Uid,
    should_update, SPEC_VERSION,
)

from neuron.submissions import get_submission

from base_validator import API_VERSION
from base_validator.metrics import BenchmarkResults, BenchmarkState, CheckpointBenchmark

from .wandb_args import add_wandb_args

VALIDATOR_VERSION = "2.6.1"
WEIGHTS_VERSION = 28

WINNER_PERCENTAGE = 0.8
BUCKET_STEP_THRESHOLD = 1.01
IMPROVEMENT_BENCHMARK_PERCENTAGE = 1.10

WinnerList: TypeAlias = list[tuple[Uid, float]]


logger = get_logger(__name__)


@dataclass
class ContestSubmissionsBucket:
    scores: WinnerList
    previous_day_winners: bool = False


def _get_incentive(rank: int, sequence_ratio: float):
    return WINNER_PERCENTAGE * (sequence_ratio ** rank)


def _winner_percentage_sequence_ratio(sample_count: int):
    if not sample_count:
        return 1 - WINNER_PERCENTAGE

    if sample_count == 1:
        return 1 / WINNER_PERCENTAGE

    polynomial = Polynomial([1 - WINNER_PERCENTAGE, -1] + ([0.0] * (sample_count - 2)) + [WINNER_PERCENTAGE])
    real_roots = [float(real(root)) for root in polynomial.roots() if isreal(root) and root >= 0.0]

    return real_roots[0]


class ContestState:
    id: ContestId
    miner_score_version: int
    submission_spec_version: int
    miner_info: list[CheckpointSubmission | None]

    def __init__(
        self,
        contest_id: ContestId,
        miner_info: list[CheckpointSubmission | None],
    ):
        self.id = contest_id
        self.miner_score_version = WEIGHTS_VERSION
        self.miner_info = miner_info

    # Backwards compatibility
    def __setstate__(self, state):
        if "miner_score_versions" in state:
            del state["miner_score_versions"]

        self.miner_score_version = state.get("miner_score_version", 0)
        self.submission_spec_version = state.get("submission_spec_version", 0)
        self.__dict__.update(state)

    def __repr__(self):
        return f"ContestState(id={self.id}, miner_score_version={self.miner_score_version}, miner_info={self.miner_info})"


class Validator:
    config: dict[str, Any]
    substrate: SubstrateInterface
    metagraph: Metagraph
    keypair: Keypair
    uid: Uid

    hotkeys: list[Key]
    step: int

    last_day: date | None
    contest_state: ContestState | None
    previous_day_winners: WinnerList
    benchmarking: bool

    wandb_run: Run | None
    wandb_run_date: date | None

    current_block: int
    last_block_fetch: datetime | None = None
    attempted_set_weights: bool = False

    benchmarks: list[CheckpointBenchmark | None]
    failed: set[int]
    contest: Contest

    websocket: ClientConnection

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        from .diagnostics import save_validator_diagnostics
        save_validator_diagnostics(self.config)

        logger.info("Setting up bittensor objects")

        self.substrate = get_substrate(
            subtensor_network=self.config["subtensor.network"],
            subtensor_address=self.config["subtensor.chain_endpoint"]
        )

        self.metagraph = Metagraph(
            self.substrate,

            netuid=self.config["netuid"],
            load_old_nodes=False,
        )

        self.metagraph.sync_nodes()

        self.keypair = load_hotkey_keypair(
            wallet_name=self.config["wallet.name"],
            hotkey_name=self.config["wallet.hotkey"],
        )

        self.hotkeys = list(self.metagraph.nodes.keys())

        hotkey = self.keypair.ss58_address

        self.uid = self.hotkeys.index(hotkey)
        self.step = 0

        self.last_day = None
        self.contest_state = None
        self.previous_day_winners = []
        self.benchmarking = False

        self.wandb_run = None
        self.wandb_run_date = None

        self.benchmarks = self.clear_benchmarks()
        self.failed = set()

        self.load_state()
        self.start_wandb_run()

        self.contest = find_contest(self.contest_state.id) if self.contest_state else CURRENT_CONTEST

        self.websocket = self.connect_to_api()
        Thread(target=self.api_logs).start()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        day = self.last_day or self.current_time().date()

        if self.wandb_run and self.wandb_run_date == day:
            return

        hotkey = self.keypair.ss58_address
        netuid = self.metagraph.netuid

        name = f"validator-{self.uid}-{day.year}-{day.month}-{day.day}"

        contest_id = self.contest_state.id if self.contest_state else CURRENT_CONTEST.id

        signing_message = f"{self.uid}:{hotkey}:{contest_id.name}"
        signature = f"0x{self.keypair.sign(signing_message).hex()}"

        self.wandb_run = wandb.init(
            name=name,
            id=name,
            resume="allow",
            mode="offline" if self.config["wandb.offline"] else "online",
            project=self.config["wandb.project_name"],
            entity=self.config["wandb.entity"],
            notes=self.config["wandb.notes"],
            config={
                "hotkey": hotkey,
                "type": "validator",
                "uid": self.uid,
                "contest": contest_id.name,
                "signature": signature,
            },
            allow_val_change=True,
            anonymous="allow",
            tags=[
                f"version_{VALIDATOR_VERSION}",
                f"sn{netuid}",
            ],
        )

        self.wandb_run_date = day

        logger.debug(f"Started a new wandb run: {name}")

    def start_wandb_run(self):
        if self.config["wandb.off"]:
            return

        if self.wandb_run:
            logger.info("New contest day, starting a new wandb run.")

            self.wandb_run.finish()

        self.new_wandb_run()

    def send_wandb_metrics(self, average_time: float | None = None, ranks: dict[Uid, tuple[int, bool]] | None = None):
        if not self.wandb_run:
            return

        logger.info("Uploading benchmarks to wandb")

        benchmark_data = {}

        submission_data = {
            str(uid): info.model_dump(exclude={"contest"})
            for uid, info in enumerate(self.contest_state.miner_info)
            if info
        }

        for uid, benchmark in enumerate(self.benchmarks):
            if not benchmark:
                continue

            miner_info = self.contest_state.miner_info[uid]
            if not miner_info:
                continue

            data = {
                "model": miner_info.repository,
                "revision": miner_info.revision,
                "baseline_generation_time": benchmark.baseline.generation_time,
                "generation_time": benchmark.model.generation_time,
                "similarity": benchmark.similarity_score,
                "baseline_size": benchmark.baseline.size,
                "size": benchmark.model.size,
                "baseline_vram_used": benchmark.baseline.vram_used,
                "vram_used": benchmark.model.vram_used,
                "baseline_watts_used": benchmark.baseline.watts_used,
                "watts_used": benchmark.model.watts_used,
                "hotkey": self.hotkeys[uid],
            }

            if ranks and uid in ranks:
                rank, multiday_winner = ranks[uid]
                data["rank"] = rank
                data["multiday_winner"] = multiday_winner

            benchmark_data[str(uid)] = data

        log_data = {
            "submissions": submission_data,
            "benchmarks": benchmark_data,
            "invalid": list(self.failed),
        }

        if average_time:
            log_data["average_benchmark_time"] = average_time

        self.wandb_run.log(data=log_data)

        logger.info(log_data)
        logger.info("Benchmarks uploaded to wandb")

    @classmethod
    def add_extra_args(cls, argument_parser: ArgumentParser):
        argument_parser.add_argument(
            "--epoch_length",
            type=int,
            help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
            default=100,
        )

        argument_parser.add_argument(
            "--benchmarker_api",
            type=str,
            help="The API route to the validator benchmarking API.",
            required=True,
        )

        argument_parser.add_argument(
            "--blacklist.coldkeys",
            type=str,
            nargs="*",
            default=[
                "5CCefwu4fFXkBorK4ETJpaijXTG3LD5J2kBb7U5aEP4eABny",
                "5GWCF5UR9nhbEXdWifRL8xiMTUJ4XV4o23L7stbptaDRHMDr",
                "5DhxiGN4MfzTbyBh7gE3ABvvp5ZavZm97RWYeJMbKjMLCg3q",
                "5HQc3J7DoFAo54Luhh39TFmnvKQcXGfW2btQiG8VzJyUc1fj",
            ],
        )

        argument_parser.add_argument(
            "--blacklist.hotkeys",
            type=str,
            nargs="*",
            default=[],
        )

        add_wandb_args(argument_parser)

    @property
    def state_path(self):
        full_path = (
            Path.home() /
            ".bittensor" /
            "miners" /
            self.config["wallet.name"] /
            self.config["wallet.hotkey"] /
            f"netuid{self.metagraph.netuid}" /
            "validator"
        )

        makedirs(full_path, exist_ok=True)

        return full_path / "state.bin"

    def save_state(self):
        """Saves the state of the validator to a file."""
        logger.info("Saving validator state.")

        # Save the state of the validator to file.
        with open(self.state_path, "wb") as file:
            dump(
                {
                    "step": self.step,
                    "hotkeys": self.hotkeys,
                    "benchmarks": self.benchmarks,
                    "failed": self.failed,
                    "last_day": self.last_day,
                    "contest_state": self.contest_state,
                    "previous_day_winners": self.previous_day_winners,
                    "benchmarking": self.benchmarking,
                },
                file,
            )

    def load_state(self):
        """Loads the state of the validator from a file."""
        logger.info("Loading validator state.")

        path = self.state_path

        if not isfile(path):
            return

        # Load the state of the validator from file.
        with open(path, "rb") as file:
            state = load(file)

        self.step = state["step"]
        self.hotkeys = state["hotkeys"]
        self.benchmarks = state.get("benchmarks", self.benchmarks)
        self.failed = state.get("failed", self.failed)
        self.last_day = state["last_day"]
        self.contest_state = state["contest_state"]
        self.previous_day_winners = (
            state.get("previous_day_winners", self.previous_day_winners) or
            self.previous_day_winners
        )
        self.benchmarking = state.get("benchmarking", self.benchmarking)

        if self.contest_state:
            if self.contest_state.miner_score_version != WEIGHTS_VERSION:
                logger.warning(
                    f"Contest state has outdated weights version: {self.contest_state.miner_score_version}, "
                    f"current version: {WEIGHTS_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()
                self.contest_state.miner_score_version = WEIGHTS_VERSION

            if self.contest_state.submission_spec_version != SPEC_VERSION:
                logger.warning(
                    f"Contest state has outdated spec version: {self.contest_state.submission_spec_version}, "
                    f"current version: {SPEC_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()

                self.benchmarking = True
                self.contest_state.miner_info = self.get_miner_submissions()
                self.contest_state.submission_spec_version = SPEC_VERSION

    def clear_benchmarks(self) -> list[CheckpointSubmission | None]:
        return [None] * len(self.metagraph.nodes)

    def reset_miner(self, uid: Uid):
        self.benchmarks[uid] = None

        if uid in self.failed:
            self.failed.remove(uid)

    def set_miner_benchmarks(self, uid: Uid, benchmark: CheckpointBenchmark | None):
        self.benchmarks[uid] = benchmark

        if not benchmark:
            self.failed.add(uid)

    def resize(self):
        new_data = self.clear_benchmarks()
        length = len(self.metagraph.nodes)
        new_data[:length] = self.benchmarks[:length]
        self.benchmarks = new_data

    def get_sorted_contestants(self) -> list[tuple[Uid, float]]:
        contestants = [
            (uid, metric_data.calculate_score())
            for uid, metric_data in enumerate(self.benchmarks)
            if metric_data
        ]

        return sorted(contestants, key=itemgetter(1))

    def check_registration(self):
        hotkey = self.keypair.ss58_address
        if hotkey not in self.hotkeys:
            logger.error(
                f"Wallet: {self.keypair} is not registered on netuid {self.metagraph.netuid}."
            )
    def metagraph_nodes(self):
        return sorted(self.metagraph.nodes.values(), key=attrgetter("node_id"))

    def sync(self):
        self.metagraph.sync_nodes()

        self.check_registration()

        if len(self.hotkeys) != len(self.metagraph.nodes):
            self.resize()

            if self.contest_state:
                new_miner_info = [None] * len(self.metagraph.nodes)
                length = len(self.hotkeys)
                new_miner_info[:length] = self.contest_state.miner_info[:length]

                self.contest_state.miner_info = new_miner_info

        nodes = self.metagraph_nodes()

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != nodes[uid].hotkey:
                # hotkey has been replaced
                self.reset_miner(uid)

                filtered_winners = [
                    (winner_uid, score)
                    for winner_uid, score in self.previous_day_winners
                    if uid != winner_uid
                ]

                self.previous_day_winners = filtered_winners

                if self.contest_state:
                    self.contest_state.miner_info[uid] = None

        self.hotkeys = list(self.metagraph.nodes.keys())

        try:
            self.set_weights()

            self.attempted_set_weights = True

            self.metagraph.sync_nodes()
        except Exception as e:
            logger.error(f"Failed to set weights", exc_info=e)

    def set_weights(self):
        if self.attempted_set_weights:
            return

        if not self.contest_state:
            logger.info("Will not set weights as the contest state has not been set")
            return

        if self.benchmarking:
            logger.info("Will not set weights as benchmarking is not done")
            return

        logger.info("Setting weights")

        buckets = [ContestSubmissionsBucket(scores) for scores in self.get_score_buckets()]

        if len(self.previous_day_winners):
            winners = self.current_winners()

            if len(winners):
                highest_score = max([score for _, score in winners])

                winner_overrides = [
                    (uid, score)
                    for uid, score in self.previous_day_winners
                    if highest_score <= score * IMPROVEMENT_BENCHMARK_PERCENTAGE
                ]

                if len(winner_overrides):
                    buckets.append(ContestSubmissionsBucket(winner_overrides, previous_day_winners=True))

        highest_bucket = len(buckets) - 1

        ranks: dict[Uid, tuple[int, bool]] = {}

        for index, bucket in enumerate(buckets):
            bucket_rank = highest_bucket - index
            for uid, _ in bucket.scores:
                ranks[uid] = (bucket_rank, bucket.previous_day_winners)

        self.send_wandb_metrics(ranks=ranks)

        sequence_ratio = _winner_percentage_sequence_ratio(len(buckets))

        weights = numpy.zeros(len(self.metagraph.nodes))

        for index, bucket in enumerate(buckets):
            bucket_incentive = _get_incentive(highest_bucket - index, sequence_ratio)

            for uid, score in bucket.scores:
                weights[uid] = bucket_incentive / len(bucket.scores)

        uids = numpy.indices(weights.shape)[0]

        set_node_weights(
            self.substrate,
            self.keypair,
            node_ids=list(uids),
            node_weights=list(weights),
            netuid=self.metagraph.netuid,
            validator_node_id=self.uid,
            version_key=WEIGHTS_VERSION,
        )

        self.metagraph.sync_nodes()

    def get_score_buckets(self) -> list[WinnerList]:
        sorted_contestants = cast(list[tuple[Uid, float]], self.get_sorted_contestants())

        buckets: list[WinnerList] = [[]]

        last_score = sorted_contestants[0][1] if len(sorted_contestants) else None

        for contestant in sorted_contestants:
            _, score = contestant

            if last_score and score > last_score * BUCKET_STEP_THRESHOLD:
                # New bucket
                buckets.append([contestant])
            else:
                buckets[len(buckets) - 1].append(contestant)

            last_score = score

        return buckets

    def current_winners(self) -> WinnerList:
        return [(uid, score) for uid, score in self.get_score_buckets()[-1] if score > 0.0]

    def get_miner_submissions(self):
        visited_repositories: dict[str, tuple[Uid, int]] = {}
        visited_revisions: dict[str, tuple[Uid, int]] = {}

        miner_info: list[CheckpointSubmission | None] = []

        for hotkey, node in tqdm(self.metagraph.nodes.items()):
            if (
                hotkey in self.config["blacklist.hotkeys"] or
                node.coldkey in self.config["blacklist.coldkeys"]
            ):
                miner_info.append(None)
                continue

            logger.info(f"Getting submission for hotkey {hotkey}")

            submission = get_submission(
                self.substrate,
                self.metagraph.netuid,
                hotkey,
            )

            if not submission:
                miner_info.append(None)
                continue

            info, block = submission

            existing_repository_submission = visited_repositories.get(info.repository)
            existing_revision_submission = visited_revisions.get(info.revision)

            if existing_repository_submission and existing_revision_submission:
                existing_submission = min(existing_repository_submission, existing_revision_submission, key=itemgetter(1))
            else:
                existing_submission = existing_repository_submission or existing_revision_submission

            if existing_submission:
                existing_uid, existing_block = existing_submission

                if block > existing_block:
                    miner_info.append(None)
                    continue

                miner_info[existing_uid] = None

            miner_info.append(info)
            visited_repositories[info.repository] = node.node_id, block
            visited_revisions[info.revision] = node.node_id, block

            time.sleep(0.2)

        return miner_info

    def connect_to_api(self):
        url: str = self.config["benchmarker_api"].replace("http", "ws")

        websocket = connect(f"{url}/logs")

        try:
            version = json.loads(websocket.recv())["version"]
        except:
            raise RuntimeError("Validator API out of date")

        if version != API_VERSION:
            raise RuntimeError(
                f"Validator API has mismatched version, received {version} but expected {API_VERSION}"
            )

        return websocket

    def api_logs(self):
        while True:
            try:
                for line in self.websocket:
                    output = sys.stderr if line.startswith("err:") else sys.stdout

                    print(f"[API] -{line[4:]}", file=output)
            except ConnectionClosedError:
                self.websocket = self.connect_to_api()

    def start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        logger.info(f"Sending {submissions} for testing")

        submissions_json = RootModel[dict[Key, CheckpointSubmission]](submissions).model_dump_json()

        api = self.config["benchmarker_api"]

        nonce = str(time.time_ns())

        signature = f"0x{self.keypair.sign(nonce).hex()}"

        state_response = requests.post(
            f"{api}/start",
            headers={
                "Content-Type": "application/json",
                "X-Nonce": nonce,
                "Signature": signature,
            },
            data=submissions_json,
        )

        state_response.raise_for_status()

    def current_time(self):
        return datetime.now(tz=ZoneInfo("America/New_York"))

    def non_tested_miners(self):
        return list(
            {
                uid
                for uid, benchmark in enumerate(self.benchmarks)
                if self.contest_state.miner_info[uid] and not benchmark and uid not in self.failed
            }
        )

    def do_step(self, block: int):
        now = self.current_time()

        if (not self.last_day or self.last_day < now.date()) and now.hour >= 12:
            # Past noon, should start collecting submissions
            logger.info("Collecting all submissions")

            miner_info = self.get_miner_submissions()

            logger.info(f"Got {miner_info} submissions")

            if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
                # New contest, restart
                logger.info(f"Working on contest {self.contest.id.name} today's submissions")

                nodes = self.metagraph_nodes()

                submissions = {
                    nodes[uid].hotkey: submission
                    for uid, submission in enumerate(miner_info)
                    if submission
                }

                self.start_benchmarking(submissions)

                self.contest = CURRENT_CONTEST

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()

                self.contest_state = ContestState(self.contest.id, miner_info)
                self.previous_day_winners = []
            else:
                updated_uids = [
                    uid
                    for uid in range(len(self.metagraph.nodes))
                    if should_update(self.contest_state.miner_info[uid], miner_info[uid])
                ] + self.non_tested_miners()

                nodes = self.metagraph_nodes()

                submissions = {
                    nodes[uid].hotkey: miner_info[uid]
                    for uid in set(updated_uids)
                    if miner_info[uid]
                }

                self.start_benchmarking(submissions)

                logger.info(f"Miners {updated_uids} changed their submissions or have not been tested yet")

                for uid in updated_uids:
                    self.reset_miner(uid)

                self.contest_state.miner_info = miner_info

                winners = self.current_winners()

                if len(winners):
                    if len(self.previous_day_winners):
                        bucket_score = min([score for _, score in self.previous_day_winners])

                        new_winners = [
                            (uid, score)
                            for uid, score in winners
                            if bucket_score <= score * IMPROVEMENT_BENCHMARK_PERCENTAGE
                        ]

                        if len(new_winners):
                            # New winner
                            self.previous_day_winners = new_winners
                    else:
                        self.previous_day_winners = winners

            self.last_day = now.date()

            self.start_wandb_run()

            self.benchmarking = True

            self.step += 1
            return

        last_update = self.metagraph.nodes[self.keypair.ss58_address].last_updated
        blocks_elapsed = block - last_update
        epoch_length = self.config["epoch_length"]

        if blocks_elapsed >= epoch_length:
            logger.info(f"{blocks_elapsed} blocks since last update, resyncing metagraph")
            self.sync()

            # Recalculate in-case weights were set
            blocks_elapsed = block - self.metagraph.nodes[self.keypair.ss58_address].last_updated
        else:
            logger.info(
                f"{blocks_elapsed} since last update, "
                f"{epoch_length - blocks_elapsed} blocks remaining until metagraph sync"
            )

        if not self.benchmarking:
            self.step += 1

            if self.contest_state:
                remaining = self.non_tested_miners()

                if len(remaining):
                    nodes = self.metagraph_nodes()

                    submissions = {
                        nodes[uid].hotkey: self.contest_state.miner_info[uid]
                        for uid in remaining
                    }

                    self.start_benchmarking(submissions)
                    self.benchmarking = True

                    self.save_state()

                    return

            blocks_to_wait = epoch_length - blocks_elapsed

            if blocks_to_wait <= 0:
                # Randomize in case multiple validators are in this same state,
                # to avoid multiple validators setting weights all in the same block
                blocks_to_wait = random.randint(1, 10)

            logger.info(f"Nothing to do in this step, sleeping for {blocks_to_wait} blocks")
            time.sleep(blocks_to_wait * 12)

            return

        api = self.config["benchmarker_api"]

        state_response = requests.get(f"{api}/state")

        state_response.raise_for_status()

        result = BenchmarkResults.model_validate(state_response.json())

        if result.state == BenchmarkState.NOT_STARTED:
            # API likely crashed or got restarted, need to re-benchmark any submissions sent to API
            logger.info(
                "API in different state than expected, likely restarted. "
                "Sending submissions again for testing"
            )

            nodes = self.metagraph_nodes()

            submissions = {
                nodes[uid].hotkey: self.contest_state.miner_info[uid]
                for uid in self.non_tested_miners()
            }

            self.start_benchmarking(submissions)

            self.step += 1

            self.save_state()
            return

        for hotkey, benchmark in result.results.items():
            logger.info(f"Updating {hotkey}'s benchmarks to {benchmark}")
            if hotkey in self.hotkeys:
                self.set_miner_benchmarks(self.hotkeys.index(hotkey), benchmark)

        self.send_wandb_metrics(average_time=result.average_benchmark_time)

        if result.state == BenchmarkState.FINISHED:
            logger.info(
                "Benchmarking API has reported submission testing as done. "
                "Miner metrics updated:"
            )
            logger.info(self.benchmarks)

            self.benchmarking = False
            self.step += 1

            self.save_state()
            return

        self.save_state()

        blocks = epoch_length / 4
        logger.info(f"Benchmarking in progress, sleeping for {blocks} blocks")
        time.sleep(blocks * 12)

        self.step += 1

    @property
    def block(self):
        if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
            self.current_block = self.substrate.get_block_number(None)  # type: ignore
            self.last_block_fetch = datetime.now()
            self.attempted_set_weights = False

        return self.current_block

    def run(self):
        while True:
            current_block = self.block

            try:
                logger.info(f"Step {self.step}, block {current_block}")

                self.do_step(current_block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
                    logger.error(f"Error during validation step {self.step}", exc_info=e)
                    continue

                raise


def main():
    Validator().run()


if __name__ == '__main__':
    main()
