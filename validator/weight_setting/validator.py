import asyncio
import random
from argparse import ArgumentParser
from asyncio import sleep
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from itertools import islice
from json import JSONDecodeError
from math import ceil
from operator import itemgetter, attrgetter
from os import makedirs
from os.path import isfile
from pathlib import Path
from pickle import dump, load
from ssl import SSLEOFError
from typing import Any

import requests
import wandb

from base_validator import BenchmarkState, BenchmarkResults, AutoUpdater
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights
from fiber.logging_utils import get_logger
from substrateinterface.exceptions import SubstrateRequestException
from substrateinterface import SubstrateInterface, Keypair
from wandb.sdk.wandb_run import Run

from neuron import (
    get_config,
    ContestId,
    CURRENT_CONTEST,
    INPUTS_ENDPOINT,
    find_contest,
    Contest,
    Key,
    Uid,
    MinerModelInfo,
    TIMEZONE,
    ModelRepositoryInfo,
    SPEC_VERSION,
    get_submissions,
    BENCHMARKS_VERSION,
    CheckpointBenchmark,
    MetricData,
)
from neuron.device import ContestDeviceValidationError
from .benchmarking_api import BenchmarkingApi, benchmarking_api
from .wandb_args import add_wandb_args
from .winner_selection import get_scores, get_contestant_scores, get_tiers, get_contestant_tier

VALIDATOR_VERSION: tuple[int, int, int] = (5, 3, 1)
VALIDATOR_VERSION_STRING = ".".join(map(str, VALIDATOR_VERSION))

WEIGHTS_VERSION = (
    VALIDATOR_VERSION[0] * 10000 +
    VALIDATOR_VERSION[1] * 100 +
    VALIDATOR_VERSION[2]
)

COLLECTED_SUBMISSIONS_VERSION = SPEC_VERSION * 10 + 2

logger = get_logger(__name__)


@dataclass
class ContestState:
    id: ContestId
    miner_score_version: int
    submission_spec_version: int
    miner_info: list[MinerModelInfo | None]

    def __init__(
        self,
        contest_id: ContestId,
        miner_info: list[MinerModelInfo | None],
    ):
        self.id = contest_id
        self.miner_score_version = BENCHMARKS_VERSION
        self.submission_spec_version = COLLECTED_SUBMISSIONS_VERSION
        self.miner_info = miner_info


class Validator:
    auto_updater: AutoUpdater
    config: dict[str, Any]
    substrate: SubstrateInterface
    metagraph: Metagraph
    keypair: Keypair
    uid: Uid

    hotkeys: list[Key]
    step: int

    last_day: date | None
    contest_state: ContestState | None
    benchmarking: bool
    benchmarking_api_urls: list[str]
    benchmarking_apis: list[BenchmarkingApi]

    wandb_run: Run | None
    wandb_run_date: date | None

    current_block: int
    last_block_fetch: datetime | None = None
    last_metagraph_sync: int = 0
    attempted_set_weights: bool = False

    benchmarks: list[CheckpointBenchmark | None]
    last_benchmarks: list[CheckpointBenchmark | None]
    baseline_metrics: MetricData | None
    average_benchmarking_time: float | None
    benchmarking_state: BenchmarkState
    invalid: dict[int, str]
    contest: Contest

    def __init__(self):
        self.auto_updater = AutoUpdater()
        self.config = get_config(Validator.add_extra_args)

        from .diagnostics import save_validator_diagnostics
        save_validator_diagnostics(self.config)

        logger.info(f"Validator version {VALIDATOR_VERSION_STRING}! Loading...")

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
        self.benchmarking = False
        self.benchmarking_api_urls = self.config["benchmarker_api"]

        self.wandb_run = None
        self.wandb_run_date = None

        self.benchmarks = self.clear_benchmarks()
        self.last_benchmarks = self.clear_benchmarks()
        self.baseline_metrics = None
        self.average_benchmarking_time = None
        self.benchmarking_state = BenchmarkState.NOT_STARTED
        self.invalid = {}

        self.load_state()
        self.start_wandb_run()

        self.contest = find_contest(self.contest_state.id) if self.contest_state else CURRENT_CONTEST

    def start_wandb_run(self):
        if self.config["wandb.off"]:
            return

        if self.wandb_run:
            logger.info("New contest day, starting a new wandb run.")
            self.wandb_run.finish()

        hotkey = self.keypair.ss58_address
        day = self.last_day or self.current_time().date()
        name = f"validator-{self.uid}-{day.year}-{day.month}-{day.day}"

        contest_id = self.contest_state.id if self.contest_state else CURRENT_CONTEST.id

        signing_message = f"{name}:{hotkey}:{contest_id.name}"
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
                f"version_{VALIDATOR_VERSION_STRING}",
                f"sn{self.metagraph.netuid}",
            ],
        )

        self.wandb_run_date = day
        self.wandb_run.log(data={"benchmarking_state": self.benchmarking_state.name})

        logger.debug(f"Started a new wandb run: {name}")

    def send_wandb_metrics(self):
        if not self.wandb_run:
            return

        submission_data = {
            str(uid): {
                "hotkey": self.hotkeys[uid],
                "repository": info.repository.url,
                "revision": info.repository.revision,
                "block": info.block,
            }
            for uid, info in enumerate(self.contest_state.miner_info)
            if info
        }

        log_data = {
            "submissions": submission_data,
            "benchmarks": self.get_wandb_benchmarks(self.benchmarks),
            "last_benchmarks": self.get_wandb_benchmarks(self.last_benchmarks),
            "invalid": self.invalid,
            "benchmarking_state": self.benchmarking_state.name,
        }

        if self.average_benchmarking_time:
            log_data["average_benchmark_time"] = self.average_benchmarking_time

        if self.baseline_metrics:
            log_data["baseline"] = self.baseline_metrics.model_dump()

        self.wandb_run.log(data=log_data)

        logger.info("Benchmarks uploaded to wandb")

    def get_wandb_benchmarks(self, benchmarks: list[CheckpointBenchmark | None]):
        benchmark_data = {}
        tiers: list[list[Uid]] = []

        if self.baseline_metrics:
            contestants = get_contestant_scores(benchmarks, self.baseline_metrics)
            tiers = get_tiers(contestants)

        for uid, benchmark in enumerate(benchmarks):
            if not benchmark:
                continue

            miner_info = self.contest_state.miner_info[uid]
            if not miner_info:
                continue

            data = {
               "similarity": benchmark.average_similarity,
               "min_similarity": benchmark.min_similarity,
            } | benchmark.model_dump()

            if self.baseline_metrics:
                data["score"] = CURRENT_CONTEST.calculate_score(self.baseline_metrics, benchmark)
            if tiers:
                data["tier"] = get_contestant_tier(tiers, uid)

            benchmark_data[str(uid)] = data

        return benchmark_data

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
            nargs="*",
            help="The API route to the validator benchmarking API.",
            required=True,
        )

        argument_parser.add_argument(
            "--delayed_weights.off",
            action="store_true",
            help="Turn off delayed weight setting.",
            default=False,
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
                    "last_benchmarks": self.last_benchmarks,
                    "baseline_benchmarks": self.baseline_metrics,
                    "average_benchmarking_time": self.average_benchmarking_time,
                    "benchmarking_state": self.benchmarking_state,
                    "invalid": self.invalid,
                    "last_day": self.last_day,
                    "contest_state": self.contest_state,
                    "benchmarking": self.benchmarking,
                    "last_metagraph_sync": self.last_metagraph_sync,
                },
                file,
            )

    def load_state(self):
        """Loads the state of the validator from a file."""
        path = self.state_path

        if not isfile(path):
            return

        logger.info("Loading validator state.")

        # Load the state of the validator from file.
        with open(path, "rb") as file:
            state = load(file)

        self.step = state["step"]
        self.hotkeys = state["hotkeys"]
        self.benchmarks = state.get("benchmarks", self.benchmarks)
        self.last_benchmarks = state.get("last_benchmarks", self.last_benchmarks)
        self.baseline_metrics = state.get("baseline_benchmarks", self.baseline_metrics)
        self.average_benchmarking_time = state.get("average_benchmarking_time", self.average_benchmarking_time)
        self.benchmarking_state = state.get("benchmarking_state", self.benchmarking_state)
        self.invalid = state.get("invalid", self.invalid)
        self.last_day = state["last_day"]
        self.contest_state = state["contest_state"]
        self.benchmarking = state.get("benchmarking", self.benchmarking)
        self.last_metagraph_sync = state.get("last_metagraph_sync", self.last_metagraph_sync)

        if self.contest_state:
            if self.contest_state.miner_score_version != BENCHMARKS_VERSION:
                logger.warning(
                    f"Contest state has outdated weights version: {self.contest_state.miner_score_version}, "
                    f"current version: {BENCHMARKS_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.invalid.clear()
                self.contest_state.miner_score_version = BENCHMARKS_VERSION

            if self.contest_state.submission_spec_version != COLLECTED_SUBMISSIONS_VERSION:
                logger.warning(
                    f"Contest state has outdated spec version: {self.contest_state.submission_spec_version}, "
                    f"current version: {COLLECTED_SUBMISSIONS_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.invalid.clear()

                self.benchmarking = True
                self.contest_state.miner_info = self.get_miner_submissions()
                self.contest_state.submission_spec_version = COLLECTED_SUBMISSIONS_VERSION

    def clear_benchmarks(self) -> list[CheckpointBenchmark | None]:
        return [None] * len(self.metagraph.nodes)

    def reset_miner(self, uid: Uid):
        self.benchmarks[uid] = None
        self.last_benchmarks[uid] = None

        if uid in self.invalid:
            del self.invalid[uid]

    def resize(self):
        new_data = self.clear_benchmarks()
        length = len(self.metagraph.nodes)
        new_data[:length] = self.benchmarks[:length]
        self.benchmarks = new_data

    def check_registration(self):
        hotkey = self.keypair.ss58_address
        if hotkey not in self.hotkeys:
            logger.error(
                f"Wallet: {self.keypair} is not registered on netuid {self.metagraph.netuid}."
            )

    def metagraph_nodes(self):
        return sorted(self.metagraph.nodes.values(), key=attrgetter("node_id"))

    def sync_chain_nodes(self, block: int):
        logger.info("Syncing metagraph")

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

                if self.contest_state:
                    self.contest_state.miner_info[uid] = None

        self.hotkeys = list(self.metagraph.nodes.keys())
        self.last_metagraph_sync = block

    def sync(self, block: int):
        if block - self.last_metagraph_sync > self.config["epoch_length"]:
            self.sync_chain_nodes(block)

        try:
            self.set_weights()

            self.attempted_set_weights = True

            self.sync_chain_nodes(block)
        except SubstrateRequestException as e:
            logger.error(f"Failed to set weights: {e}")
        except Exception as e:
            logger.error(f"Failed to set weights", exc_info=e)

    def set_weights(self):
        if self.attempted_set_weights:
            return

        equal_weights = False
        delayed_weights = not self.config["delayed_weights.off"]
        benchmarks = self.last_benchmarks if delayed_weights else self.benchmarks

        if not delayed_weights and self.benchmarking:
            logger.info("Not setting new weights as benchmarking is not done, reusing old weights")
            delayed_weights = True
            benchmarks = self.last_benchmarks

        if not self.contest_state:
            logger.info("Will not set new weights as the contest state has not been set, setting to all ones")
            equal_weights = True

        elif not self.baseline_metrics:
            logger.info("Will not set new weights as the baseline benchmarks have not been set, setting to all ones")
            equal_weights = True

        elif all(benchmark is None for benchmark in self.last_benchmarks):
            if any(benchmark is not None for benchmark in self.benchmarks):
                logger.info("Setting weights to current benchmarks as the previous day's benchmarks have not been set")
                self.last_benchmarks = self.benchmarks
            elif delayed_weights:
                logger.info("Will not set new weights as the previous day's benchmarks have not been set, setting to all ones")
                equal_weights = True

        if equal_weights:
            uids = list(range(len(self.metagraph.nodes)))
            weights = [1.0] * len(self.metagraph.nodes)

            set_node_weights(
                self.substrate,
                self.keypair,
                node_ids=list(uids),
                node_weights=list(weights),
                netuid=self.metagraph.netuid,
                validator_node_id=self.uid,
                version_key=WEIGHTS_VERSION,
            )

            return

        logger.info("Setting weights")

        blacklisted_keys = self.get_blacklisted_keys()
        for hotkey, node in self.metagraph.nodes.items():
            uid = self.hotkeys.index(hotkey)
            if benchmarks[uid]:
                if self.is_blacklisted(blacklisted_keys, hotkey, node.coldkey):
                    logger.warning(f"Not setting weights for blacklisted hotkey {hotkey}")
                    self.reset_miner(uid)
                elif not self.contest_state.miner_info[uid]:
                    logger.warning(f"Not setting weights for hotkey {hotkey} as their submission was not found")
                    self.reset_miner(uid)

        contestants = get_contestant_scores(benchmarks, self.baseline_metrics)
        tiers = get_tiers(contestants)
        blocks = [info.block if info else None for info in self.contest_state.miner_info]
        weights = get_scores(tiers, blocks, len(self.metagraph.nodes))

        self.send_wandb_metrics()

        if sum(weights) <= 0.0:
            weights = [1.0] * len(self.metagraph.nodes)

        set_node_weights(
            self.substrate,
            self.keypair,
            node_ids=list(range(len(self.metagraph.nodes))),
            node_weights=weights,
            netuid=self.metagraph.netuid,
            validator_node_id=self.uid,
            version_key=WEIGHTS_VERSION,
        )

        self.metagraph.sync_nodes()

    @staticmethod
    def get_blacklisted_keys():
        response = requests.get(
            f"{INPUTS_ENDPOINT}/blacklist", headers={
                "Content-Type": "application/json"
            },
        )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def is_blacklisted(blacklisted_keys: dict, hotkey: str, coldkey: str):
        return hotkey in blacklisted_keys["hotkeys"] or coldkey in blacklisted_keys["coldkeys"]

    def get_miner_submissions(self) -> list[MinerModelInfo | None]:
        blacklisted_keys = self.get_blacklisted_keys()

        hotkeys = [hotkey for hotkey, node in self.metagraph.nodes.items() if not self.is_blacklisted(blacklisted_keys, hotkey, node.coldkey)]

        return get_submissions(
            substrate=self.substrate,
            hotkeys=hotkeys,
            netuid=self.metagraph.netuid,
            block=self.block,
        )

    async def send_submissions_to_api(self, apis: list[BenchmarkingApi], submissions: dict[Key, ModelRepositoryInfo]):
        iterator = iter(submissions.items())

        chunk_size = ceil(len(submissions) / len(apis))

        chunks = [
            (api, list(islice(iterator, chunk_size)))
            for api in apis
        ]

        await asyncio.gather(
            *[
                api.start_benchmarking(dict(chunk))
                for api, chunk in chunks
            ],
        )

    def start_benchmarking(self, submissions: dict[Key, ModelRepositoryInfo]):
        return self.send_submissions_to_api(self.benchmarking_apis, submissions)

    @staticmethod
    def current_time():
        return datetime.now(tz=TIMEZONE)

    def non_tested_miners(self) -> list[Uid]:
        return list(
            {
                uid
                for uid, benchmark in enumerate(self.benchmarks)
                if self.contest_state.miner_info[uid] and not benchmark and uid not in self.invalid
            }
        )

    def initialize_contest(self, now: datetime):
        logger.info("Collecting all submissions")

        miner_info = self.get_miner_submissions()

        logger.info(f"Got {len([info for info in miner_info if info])} submissions")

        logger.info(f"Working on contest {self.contest.id.name}")

        if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
            # New contest, restart
            self.contest = CURRENT_CONTEST
            self.contest_state = ContestState(self.contest.id, miner_info)
        else:
            self.contest_state.miner_info = miner_info

        self.average_benchmarking_time = None
        self.benchmarking_state = BenchmarkState.NOT_STARTED

        logger.info(f"Setting updated benchmarks")
        self.last_benchmarks = self.benchmarks

        self.benchmarks = self.clear_benchmarks()
        self.invalid.clear()

        self.last_day = now.date()

        self.start_wandb_run()

        self.benchmarking = False

        self.step += 1

    async def do_step(self, block: int):
        now = self.current_time()

        if (not self.last_day or self.last_day < now.date()) and now.hour >= 12:
            # Past noon, should start collecting submissions
            self.initialize_contest(now)
            return

        last_update = self.metagraph.nodes[self.keypair.ss58_address].last_updated
        blocks_elapsed = block - last_update
        epoch_length = self.config["epoch_length"]

        if blocks_elapsed >= epoch_length:
            logger.info(f"{blocks_elapsed} blocks since weight setting, attempting to set weights")
            self.sync(block)

            # Recalculate in-case weights were set
            blocks_elapsed = block - self.metagraph.nodes[self.keypair.ss58_address].last_updated
        else:
            logger.info(
                f"{blocks_elapsed} since last update, "
                f"{epoch_length - blocks_elapsed} blocks remaining until weight setting"
            )

        if not self.benchmarking:
            self.step += 1

            if self.contest_state:
                remaining = self.non_tested_miners()

                if remaining:
                    nodes = self.metagraph_nodes()

                    submissions = {
                        nodes[uid].hotkey: self.contest_state.miner_info[uid].repository
                        for uid in remaining
                    }

                    try:
                        await self.start_benchmarking(submissions)
                    except Exception as e:
                        logger.error(f"Failed to start benchmarking, retrying in 60 seconds", exc_info=e)
                        await sleep(60)
                        return

                    self.benchmarking = True

                    self.save_state()

                    return

            blocks_to_wait = epoch_length - blocks_elapsed

            if blocks_to_wait <= 0:
                # Randomize in case multiple validators are in this same state,
                # to avoid multiple validators setting weights all in the same block
                blocks_to_wait = random.randint(1, 10)

            await self.sleep_for_blocks(now, blocks_to_wait, "Nothing to do in this step")

            return

        states: tuple[BenchmarkResults] = await asyncio.gather(
            *[
                api.state()
                for api in self.benchmarking_apis
            ],
        )

        not_started = []
        in_progress = []
        finished = []

        for index, result in enumerate(states):
            match result.state:
                case BenchmarkState.NOT_STARTED:
                    not_started.append((index, result))
                case BenchmarkState.IN_PROGRESS:
                    in_progress.append((index, result))
                case BenchmarkState.FINISHED:
                    finished.append((index, result))

            if result.baseline_metrics and self.baseline_metrics != result.baseline_metrics:
                self.baseline_metrics = result.baseline_metrics
                logger.info(f"Updated baseline benchmarks to {result.baseline_metrics}")

        self.benchmarking_state = min((result.state for result in states), key=lambda state: state.value)

        with_results = in_progress + finished

        if not_started:
            api_indices = list(map(itemgetter(0), not_started))

            api_names = ",".join(
                str(index + 1)
                for index in api_indices
            )

            # API likely crashed or got restarted, need to re-benchmark any submissions sent to API
            logger.info(
                f"APIs {api_names} are in a different state than expected, likely restarted. "
                "Sending submissions again for testing"
            )

            nodes = self.metagraph_nodes()

            submissions = {
                nodes[uid].hotkey: self.contest_state.miner_info[uid].repository
                for uid in self.non_tested_miners()
            }

            apis = [
                self.benchmarking_apis[index]
                for index in api_indices
            ]

            await self.send_submissions_to_api(apis, submissions)

            if not with_results:
                self.step += 1
                self.save_state()

                return

        benchmark_times = [
            result.average_benchmark_time
            for _, result in with_results
            if result.average_benchmark_time
        ]

        for _, result in with_results:
            def get_uid(hotkey: Key) -> Uid | None:
                if not hotkey in self.hotkeys:
                    logger.info(f"{hotkey} not found, skipping")
                    return None

                uid = self.hotkeys.index(hotkey)

                if not self.contest_state.miner_info[uid]:
                    logger.info(f"{hotkey} has no submission, skipping")
                    return None

                return uid


            for hotkey, benchmark in result.results.items():
                uid = get_uid(hotkey)
                if uid is None:
                    continue

                if benchmark and self.benchmarks[uid] != benchmark:
                    logger.info(f"Updating {hotkey}'s benchmarks to {benchmark}")
                self.benchmarks[uid] = benchmark

            for hotkey, error_message in result.invalid.items():
                uid = get_uid(hotkey)
                if uid is None:
                    continue

                if error_message and error_message != self.invalid.get(uid):
                    logger.info(f"Marking {hotkey}'s submission as invalid: '{error_message}'")
                self.invalid[uid] = error_message

        self.average_benchmarking_time = (sum(benchmark_times) / len(benchmark_times)) if benchmark_times else None
        self.send_wandb_metrics()

        if not not_started and not in_progress and finished:
            logger.info("Benchmarking APIs have reported submission testing as done.")

            self.benchmarking = False
            self.step += 1

            self.save_state()
            return

        self.step += 1

        self.save_state()

        await self.sleep_for_blocks(now, epoch_length / 4, "Benchmarking in progress")

    async def sleep_for_blocks(self, now: datetime, blocks: int, reason: str):
        next_noon = datetime.combine(now.date() + timedelta(days=int(now.hour >= 12)), time(12), tzinfo=TIMEZONE)
        blocks_to_sleep = min(blocks, ceil((next_noon - now).total_seconds() / 12))
        logger.info(f"{reason}, sleeping for {blocks_to_sleep} blocks")
        await sleep(blocks_to_sleep * 12)

    @property
    def block(self):
        if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
            self.current_block = self.substrate.get_block_number(None)  # type: ignore
            self.last_block_fetch = datetime.now()
            self.attempted_set_weights = False

        return self.current_block

    async def run(self):
        self.benchmarking_apis = [
            benchmarking_api(self.keypair, api, index).build()
            for index, api in enumerate(self.benchmarking_api_urls)
        ]

        while True:
            try:
                current_block = self.block

                logger.info(f"Step {self.step}, block {current_block}")

                await self.do_step(current_block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
                    if isinstance(e, (SSLEOFError, JSONDecodeError)):
                        logger.error(f"Error during validation step {self.step}: {e}")
                    else:
                        logger.error(f"Error during validation step {self.step}", exc_info=e)

                    self.substrate = get_substrate(subtensor_address=self.substrate.url)

                    continue

                for api in self.benchmarking_apis:
                    await api.close()

                raise


def main():
    asyncio.run(Validator().run())


if __name__ == '__main__':
    main()
