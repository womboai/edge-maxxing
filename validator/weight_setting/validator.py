import asyncio
import random
import time
from argparse import ArgumentParser
from datetime import date, datetime
from itertools import islice
from math import ceil
from operator import itemgetter, attrgetter
from os import makedirs
from os.path import isfile
from pathlib import Path
from pickle import dump, load
from typing import Any

import requests
import wandb
from base_validator.hash import load_image_hash
from base_validator.metrics import BenchmarkState, CheckpointBenchmark, BenchmarkResults, MetricData
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights, get_weights_set_by_node
from fiber.logging_utils import get_logger
from substrateinterface import SubstrateInterface, Keypair
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
from weight_setting.deduplication import find_duplicates

from neuron import (
    get_config,
    ContestId,
    CURRENT_CONTEST,
    INPUTS_ENDPOINT,
    find_contest,
    ContestDeviceValidationError,
    Contest,
    Key,
    Uid,
    MinerModelInfo,
    TIMEZONE,
    ModelRepositoryInfo, SPEC_VERSION,
)
from neuron.submissions import get_submission
from .benchmarking_api import BenchmarkingApi, benchmarking_api
from .wandb_args import add_wandb_args
from .winner_selection import get_scores, get_contestant_scores

VALIDATOR_VERSION: tuple[int, int, int] = (4, 0, 0)
VALIDATOR_VERSION_STRING = ".".join(map(str, VALIDATOR_VERSION))

BENCHMARKS_VERSION = 3

WEIGHTS_VERSION = (
    VALIDATOR_VERSION[0] * 10000 +
    VALIDATOR_VERSION[1] * 100 +
    VALIDATOR_VERSION[2]
)

COLLECTED_SUBMISSIONS_VERSION = SPEC_VERSION * 10 + 1

logger = get_logger(__name__)


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
    baseline_metrics: MetricData | None
    failed: set[int]
    hash_prompt: str
    hash_seed: int
    contest: Contest

    def __init__(self):
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
        self.baseline_metrics = None
        self.failed = set()

        self.load_state()
        self.start_wandb_run()

        self.contest = find_contest(self.contest_state.id) if self.contest_state else CURRENT_CONTEST

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        day = self.last_day or self.current_time().date()

        if self.wandb_run and self.wandb_run_date == day:
            return

        hotkey = self.keypair.ss58_address
        netuid = self.metagraph.netuid

        name = f"validator-{self.uid}-{day.year}-{day.month}-{day.day}"

        contest_id = self.contest_state.id if self.contest_state else CURRENT_CONTEST.id

        signing_message = f"{name}:{hotkey}:{contest_id.name}"
        signature = f"0x{self.keypair.sign(signing_message).hex()}"

        self.wandb_run = wandb.init(
            name=name,
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

    def send_wandb_metrics(self, average_time: float | None = None):
        if not self.wandb_run:
            return

        logger.info("Uploading benchmarks to wandb")

        benchmark_data = {}

        submission_data = {
            str(uid): {
                "repository": info.repository.url,
                "revision": info.repository.revision,
                "block": info.block,
            }
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
                "generation_time": benchmark.model.generation_time,
                "similarity": benchmark.similarity_score,
                "size": benchmark.model.size,
                "vram_used": benchmark.model.vram_used,
                "watts_used": benchmark.model.watts_used,
                "hotkey": self.hotkeys[uid],
            }

            if self.baseline_metrics:
                data["score"] = benchmark.calculate_score(self.baseline_metrics)

            benchmark_data[str(uid)] = data

        log_data = {
            "submissions": submission_data,
            "benchmarks": benchmark_data,
            "invalid": list(self.failed),
        }

        if average_time:
            log_data["average_benchmark_time"] = average_time

        if self.baseline_metrics:
            log_data["baseline"] = {
                "generation_time": self.baseline_metrics.generation_time,
                "size": self.baseline_metrics.size,
                "vram_used": self.baseline_metrics.vram_used,
                "watts_used": self.baseline_metrics.watts_used,
            }

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
            nargs="*",
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
                    "baseline_benchmarks": self.baseline_metrics,
                    "failed": self.failed,
                    "last_day": self.last_day,
                    "contest_state": self.contest_state,
                    "benchmarking": self.benchmarking,
                    "last_metagraph_sync": self.last_metagraph_sync,
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
        self.baseline_metrics = state.get("baseline_benchmarks", self.baseline_metrics)
        self.failed = state.get("failed", self.failed)
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
                self.failed.clear()
                self.contest_state.miner_score_version = BENCHMARKS_VERSION

            if self.contest_state.submission_spec_version != COLLECTED_SUBMISSIONS_VERSION:
                logger.warning(
                    f"Contest state has outdated spec version: {self.contest_state.submission_spec_version}, "
                    f"current version: {COLLECTED_SUBMISSIONS_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()

                self.benchmarking = True
                self.contest_state.miner_info = self.get_miner_submissions()
                self.contest_state.submission_spec_version = COLLECTED_SUBMISSIONS_VERSION

    def clear_benchmarks(self) -> list[CheckpointBenchmark | None]:
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
        except Exception as e:
            logger.error(f"Failed to set weights", exc_info=e)

    def set_weights(self):
        if self.attempted_set_weights:
            return

        if not self.contest_state:
            logger.info("Will not set weights as the contest state has not been set")
            return

        if not self.baseline_metrics:
            logger.info("Will not set weights as the baseline benchmarks have not been set")
            return

        if self.benchmarking:
            logger.info("Not setting new weights as benchmarking is not done, reusing old weights")

            zipped_weights = get_weights_set_by_node(self.substrate, self.metagraph.netuid, self.uid, self.block)

            if not zipped_weights:
                return

            uids = map(itemgetter(0), zipped_weights)
            weights = map(itemgetter(1), zipped_weights)
            weights = map(float, weights)

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

        weights = get_scores(get_contestant_scores(self.benchmarks, self.baseline_metrics), len(self.metagraph.nodes))

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

    def get_miner_submissions(self):
        visited_repositories: dict[str, tuple[Uid, int]] = {}
        visited_revisions: dict[str, tuple[Uid, int]] = {}
        blacklisted_keys = self.get_blacklisted_keys()

        miner_info: list[MinerModelInfo | None] = []

        for hotkey, node in tqdm(self.metagraph.nodes.items()):
            if (
                hotkey in blacklisted_keys["hotkeys"] or
                node.coldkey in blacklisted_keys["coldkeys"]
            ):
                miner_info.append(None)
                continue

            logger.info(f"Getting submission for hotkey {hotkey}")

            info = get_submission(
                self.substrate,
                self.metagraph.netuid,
                hotkey,
            )

            if not info:
                miner_info.append(None)
                continue

            existing_repository_submission = visited_repositories.get(info.repository.url)
            existing_revision_submission = visited_revisions.get(info.repository.revision)

            if existing_repository_submission and existing_revision_submission:
                existing_submission = min(
                    existing_repository_submission, existing_revision_submission, key=itemgetter(1)
                )
            else:
                existing_submission = existing_repository_submission or existing_revision_submission

            if existing_submission:
                existing_uid, existing_block = existing_submission

                if info.block > existing_block:
                    miner_info.append(None)
                    continue

                miner_info[existing_uid] = None

            miner_info.append(info)
            visited_repositories[info.repository.url] = node.node_id, info.block
            visited_revisions[info.repository.revision] = node.node_id, info.block

            time.sleep(0.2)

        return miner_info

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

    def non_tested_miners(self):
        return list(
            {
                uid
                for uid, benchmark in enumerate(self.benchmarks)
                if self.contest_state.miner_info[uid] and not benchmark and uid not in self.failed
            }
        )

    async def do_step(self, block: int):
        now = self.current_time()

        if (not self.last_day or self.last_day < now.date()) and now.hour >= 12:
            # Past noon, should start collecting submissions
            logger.info("Collecting all submissions")

            miner_info = self.get_miner_submissions()

            logger.info(f"Got {miner_info} submissions")

            nodes = self.metagraph_nodes()

            logger.info(f"Working on contest {self.contest.id.name} today's submissions")

            submissions = {
                nodes[uid].hotkey: submission.repository
                for uid, submission in enumerate(miner_info)
                if submission
            }

            await self.start_benchmarking(submissions)

            self.benchmarks = self.clear_benchmarks()
            self.failed.clear()

            if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
                # New contest, restart
                self.contest = CURRENT_CONTEST

                self.contest_state = ContestState(self.contest.id, miner_info)
            else:
                self.contest_state.miner_info = miner_info

            self.last_day = now.date()

            self.start_wandb_run()

            self.benchmarking = True

            self.step += 1
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

                if len(remaining):
                    nodes = self.metagraph_nodes()

                    submissions = {
                        nodes[uid].hotkey: self.contest_state.miner_info[uid].repository
                        for uid in remaining
                    }

                    await self.start_benchmarking(submissions)
                    self.benchmarking = True

                    self.save_state()

                    return

            blocks_to_wait = epoch_length - blocks_elapsed

            if blocks_to_wait <= 0:
                # Randomize in case multiple validators are in this same state,
                # to avoid multiple validators setting weights all in the same block
                blocks_to_wait = random.randint(1, 10)

            logger.info(f"Nothing to do in this step, sleeping for {blocks_to_wait} blocks")
            await asyncio.sleep(blocks_to_wait * 12)

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
            for hotkey, benchmark in result.results.items():
                logger.info(f"Updating {hotkey}'s benchmarks to {benchmark}")
                if hotkey in self.hotkeys:
                    self.set_miner_benchmarks(self.hotkeys.index(hotkey), benchmark)

        average_time = (sum(benchmark_times) / len(benchmark_times)) if benchmark_times else None

        self.send_wandb_metrics(average_time=average_time)

        if not not_started and not in_progress:
            logger.info(
                "Benchmarking APIs have reported submission testing as done. "
                "Miner metrics updated:"
            )
            logger.info(self.benchmarks)

            benchmark_duplicate_info = [
                (load_image_hash(benchmark.image_hash), self.contest_state.miner_info[uid].block) if benchmark else None
                for uid, benchmark in enumerate(self.benchmarks)
            ]

            for duplicate_uid in find_duplicates(benchmark_duplicate_info):
                self.benchmarks[duplicate_uid] = None
                self.failed.add(duplicate_uid)

            self.benchmarking = False
            self.step += 1

            self.save_state()
            return

        self.step += 1

        self.save_state()

        blocks = epoch_length / 4
        logger.info(f"Benchmarking in progress, sleeping for {blocks} blocks")
        await asyncio.sleep(blocks * 12)

    @property
    def block(self):
        if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
            self.current_block = self.substrate.get_block_number(None)  # type: ignore
            self.last_block_fetch = datetime.now()
            self.attempted_set_weights = False

        return self.current_block

    async def run(self):
        self.benchmarking_apis = list(
            await asyncio.gather(
                *[
                    benchmarking_api(self.keypair, api, index)
                    for index, api in enumerate(self.benchmarking_api_urls)
                ],
            )
        )

        while True:
            try:
                current_block = self.block

                logger.info(f"Step {self.step}, block {current_block}")

                await self.do_step(current_block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
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
