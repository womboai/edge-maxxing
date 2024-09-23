import json
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date, datetime
from operator import itemgetter
from os import makedirs
from os.path import isfile, expanduser, join
from threading import Thread
from typing import cast, TypeAlias
from zoneinfo import ZoneInfo

import numpy
import requests
import wandb
from bittensor.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from numpy import real, isreal
from numpy.polynomial import Polynomial
from pickle import dump, load

from pydantic import RootModel
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
from websockets import ConnectionClosedError
from websockets.sync.client import connect, ClientConnection

from neuron import (
    bt,
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

VALIDATOR_VERSION = "2.3.5"
WEIGHTS_VERSION = 27

WINNER_PERCENTAGE = 0.8
BUCKET_STEP_THRESHOLD = 1.01
IMPROVEMENT_BENCHMARK_PERCENTAGE = 1.10

WinnerList: TypeAlias = list[tuple[Uid, float]]


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
    config: bt.config
    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
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
    log_thread: Thread

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        if not self.config.benchmarker_api:
            raise ValueError("--benchmarker_api required")

        from .diagnostics import save_validator_diagnostics
        save_validator_diagnostics(self.config)

        bt.logging.info("Setting up bittensor objects")

        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = bt.wallet(config=self.config)

        self.hotkeys = self.metagraph.hotkeys
        hotkey = self.wallet.hotkey.ss58_address
        if hotkey not in self.hotkeys:
            bt.logging.error(f"Hotkey '{hotkey}' has not been registered in SN{self.config.netuid}!")
            exit(1)

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

        self.log_thread = Thread(target=self.api_logs)
        self.log_thread.start()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        day = self.last_day or self.current_time().date()

        if self.wandb_run and self.wandb_run_date == day:
            return

        hotkey = self.wallet.hotkey.ss58_address
        uid = self.metagraph.hotkeys.index(hotkey)
        netuid = self.metagraph.netuid

        name = f"validator-{uid}-{day.year}-{day.month}-{day.day}"

        contest_id = self.contest_state.id if self.contest_state else CURRENT_CONTEST.id

        signing_message = f"{uid}:{hotkey}:{contest_id.name}"
        signature = f"0x{self.wallet.hotkey.sign(signing_message).hex()}"

        self.wandb_run = wandb.init(
            name=name,
            id=name,
            resume="allow",
            mode="offline" if self.config.wandb.offline else "online",
            project=self.config.wandb.project_name,
            entity=self.config.wandb.entity,
            notes=self.config.wandb.notes,
            config={
                "hotkey": hotkey,
                "type": "validator",
                "uid": uid,
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

        bt.logging.debug(f"Started a new wandb run: {name}")

    def start_wandb_run(self):
        if self.config.wandb.off:
            return

        if self.wandb_run:
            bt.logging.info("New contest day, starting a new wandb run.")

            self.wandb_run.finish()

        self.new_wandb_run()

    def send_wandb_metrics(self, ranks: dict[Uid, tuple[int, bool]] | None = None):
        if not self.wandb_run:
            return

        bt.logging.info("Uploading benchmarks to wandb")

        log_data = {}

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
                rank, multiday_winner = ranks.get(uid)
                data["rank"] = rank
                data["multiday_winner"] = multiday_winner

            log_data[str(uid)] = data

        self.wandb_run.log(data=log_data)

        bt.logging.info(log_data)
        bt.logging.info("Benchmarks uploaded to wandb")

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
        )

        argument_parser.add_argument(
            "--blacklist.coldkeys",
            type=str,
            nargs="*",
            default=["5CCefwu4fFXkBorK4ETJpaijXTG3LD5J2kBb7U5aEP4eABny"],
        )

        argument_parser.add_argument(
            "--blacklist.hotkeys",
            type=str,
            nargs="*",
            default=[],
        )

        add_wandb_args(argument_parser)

    def state_path(self):
        full_path = expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey,
                self.config.netuid,
                "validator",
            )
        )

        makedirs(full_path, exist_ok=True)

        return join(full_path, "state.bin")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        with open(self.state_path(), "wb") as file:
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
        bt.logging.info("Loading validator state.")

        path = self.state_path()

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
                bt.logging.warning(
                    f"Contest state has outdated weights version: {self.contest_state.miner_score_version}, "
                    f"current version: {WEIGHTS_VERSION}. Resetting benchmarks."
                )

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()
                self.contest_state.miner_score_version = WEIGHTS_VERSION

            if self.contest_state.submission_spec_version != SPEC_VERSION:
                bt.logging.warning(
                    f"Contest state has outdated spec version: {self.contest_state.submission_spec_version}, "
                    f"current version: {SPEC_VERSION}. Resetting benchmarks."
                )

                miner_info = self.get_miner_submissions()

                submissions = {
                    self.metagraph.hotkeys[uid]: submission
                    for uid, submission in enumerate(miner_info)
                    if submission
                }

                self.start_benchmarking(submissions)

                self.benchmarks = self.clear_benchmarks()
                self.failed.clear()

                self.benchmarking = True
                self.contest_state.miner_info = miner_info
                self.contest_state.submission_spec_version = SPEC_VERSION

    def clear_benchmarks(self) -> list[CheckpointSubmission | None]:
        return [None] * self.metagraph.n.item()

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
        length = len(self.metagraph.hotkeys)
        new_data[:length] = self.benchmarks[:length]
        self.benchmarks = new_data

    def get_sorted_contestants(self) -> list[tuple[Uid, float]]:
        contestants = [
            (uid, metric_data.calculate_score())
            for uid, metric_data in enumerate(self.benchmarks)
            if metric_data
        ]

        return sorted(contestants, key=itemgetter(1))

    def sync(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
            block=self.current_block,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )

            exit()

        self.metagraph.sync(
            subtensor=self.subtensor,
            block=self.current_block
        )

        if len(self.hotkeys) != len(self.metagraph.hotkeys):
            self.resize()

            if self.contest_state:
                new_miner_info = [None] * self.metagraph.n.item()
                length = len(self.hotkeys)
                new_miner_info[:length] = self.contest_state.miner_info[:length]

                self.contest_state.miner_info = new_miner_info

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
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

        self.hotkeys = self.metagraph.hotkeys

        try:
            self.set_weights()
        except Exception as e:
            bt.logging.error(f"Failed to set weights", exc_info=e)

    def set_weights(self):
        if self.attempted_set_weights:
            return

        if not self.contest_state:
            bt.logging.info("Will not set weights as the contest state has not been set")
            return

        if self.benchmarking:
            bt.logging.info("Will not set weights as benchmarking is not done")
            return

        bt.logging.info("Setting weights")

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

        self.send_wandb_metrics(ranks)

        sequence_ratio = _winner_percentage_sequence_ratio(len(buckets))

        weights = numpy.zeros(self.metagraph.n)

        for index, bucket in enumerate(buckets):
            bucket_incentive = _get_incentive(highest_bucket - index, sequence_ratio)

            for uid, score in bucket.scores:
                weights[uid] = bucket_incentive / len(bucket.scores)

        uids = numpy.indices(weights.shape)[0]

        bt.logging.debug("raw_weights", weights)
        bt.logging.debug("raw_weight_uids", uids)
        # Process the raw weights to final_weights via subtensor limitations.

        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        result, message = self.subtensor.set_weights(
            self.wallet,
            self.metagraph.netuid,
            uint_uids,
            uint_weights,
            WEIGHTS_VERSION,
        )

        if result:
            bt.logging.info(f"set_weights successful, {message}")
        else:
            bt.logging.warning(f"set_weights failed, {message}")

        self.attempted_set_weights = True

        self.metagraph.sync(
            subtensor=self.subtensor,
            block=self.current_block
        )

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

        for uid in tqdm(range(self.metagraph.n.item())):
            hotkey = self.metagraph.hotkeys[uid]

            if (
                hotkey in self.config.blacklist.hotkeys or
                self.metagraph.coldkeys[uid] in self.config.blacklist.coldkeys
            ):
                miner_info.append(None)
                continue

            error: Exception | None = None

            for attempt in range(3):
                try:
                    if attempt:
                        bt.logging.warning(f"Failed to get submission, attempt #{attempt + 1}")
                    else:
                        bt.logging.info(f"Getting submission for hotkey {hotkey}")

                    submission = get_submission(
                        self.subtensor,
                        self.metagraph,
                        hotkey,
                        self.current_block,
                    )

                    break
                except Exception as e:
                    error = e
                    time.sleep(0.1)
                    continue
            else:
                raise error

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
            visited_repositories[info.repository] = uid, block
            visited_revisions[info.revision] = uid, block

            time.sleep(0.2)

        return miner_info

    def connect_to_api(self):
        url: str = self.config.benchmarker_api.replace("http", "ws")

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

                    print(line[4:], file=output)
            except ConnectionClosedError:
                self.websocket = self.connect_to_api()

    def start_benchmarking(self, submissions: dict[Key, CheckpointSubmission]):
        bt.logging.info(f"Sending {submissions} for testing")

        submissions_json = RootModel[dict[Key, CheckpointSubmission]](submissions).model_dump_json()

        state_response = requests.post(
            f"{self.config.benchmarker_api}/start",
            headers={"Content-Type": "application/json"},
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
            bt.logging.info("Collecting all submissions")

            miner_info = self.get_miner_submissions()

            bt.logging.info(f"Got {miner_info} submissions")

            if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
                # New contest, restart
                bt.logging.info(f"Working on contest {self.contest.id.name} today's submissions")

                submissions = {
                    self.metagraph.hotkeys[uid]: submission
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
                    for uid in range(self.metagraph.n.item())
                    if should_update(self.contest_state.miner_info[uid], miner_info[uid])
                ] + self.non_tested_miners()

                submissions = {
                    self.metagraph.hotkeys[uid]: miner_info[uid]
                    for uid in set(updated_uids)
                    if miner_info[uid]
                }

                self.start_benchmarking(submissions)

                bt.logging.info(f"Miners {updated_uids} changed their submissions or have not been tested yet")

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

        last_update = self.metagraph.last_update[self.uid]
        blocks_elapsed = block - last_update

        if blocks_elapsed >= self.config.epoch_length:
            bt.logging.info(f"{blocks_elapsed} blocks since last update, resyncing metagraph")
            self.sync()

            # Recalculate in-case weights were set
            blocks_elapsed = block - self.metagraph.last_update[self.uid]
        else:
            bt.logging.info(
                f"{blocks_elapsed} since last update, "
                f"{self.config.epoch_length - blocks_elapsed} blocks remaining until metagraph sync"
            )

        if not self.benchmarking:
            self.step += 1

            if self.contest_state:
                remaining = self.non_tested_miners()

                if not remaining:
                    submissions = {
                        self.metagraph.hotkeys[uid]: self.contest_state.miner_info[uid]
                        for uid in remaining
                    }

                    self.start_benchmarking(submissions)
                    self.benchmarking = True

                    self.save_state()

                    return

            blocks_to_wait = self.config.epoch_length - blocks_elapsed

            if blocks_to_wait <= 0:
                # Randomize in case multiple validators are in this same state,
                # to avoid multiple validators setting weights all in the same block
                blocks_to_wait = random.randint(1, 10)

            bt.logging.info(f"Nothing to do in this step, sleeping for {blocks_to_wait} blocks")
            time.sleep(blocks_to_wait * 12)

            return

        state_response = requests.get(f"{self.config.benchmarker_api}/state")

        state_response.raise_for_status()

        result = BenchmarkResults.model_validate(state_response.json())

        if result.state == BenchmarkState.NOT_STARTED:
            # API likely crashed or got restarted, need to re-benchmark any submissions sent to API
            bt.logging.info(
                "API in different state than expected, likely restarted. "
                "Sending submissions again for testing"
            )

            submissions = {
                self.metagraph.hotkeys[uid]: self.contest_state.miner_info[uid]
                for uid in self.non_tested_miners()
            }

            self.start_benchmarking(submissions)

            self.step += 1

            self.save_state()
            return

        for hotkey, benchmark in result.results.items():
            bt.logging.info(f"Updating {hotkey}'s benchmarks to {benchmark}")
            if hotkey in self.metagraph.hotkeys:
                self.set_miner_benchmarks(self.metagraph.hotkeys.index(hotkey), benchmark)

        self.send_wandb_metrics()

        if result.state == BenchmarkState.FINISHED:
            bt.logging.info(
                "Benchmarking API has reported submission testing as done. "
                "Miner metrics updated:"
            )
            bt.logging.info(self.benchmarks)

            self.benchmarking = False
            self.step += 1

            self.save_state()
            return

        self.save_state()

        blocks_to_wait = self.config.epoch_length * 3
        bt.logging.info(f"Benchmarking in progress, sleeping for {blocks_to_wait} blocks")
        time.sleep(blocks_to_wait * 12)

        self.step += 1

    def run(self):
        while True:
            if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
                self.current_block = self.subtensor.get_current_block()
                self.last_block_fetch = datetime.now()
                self.attempted_set_weights = False

            try:
                bt.logging.info(f"Step {self.step}, block {self.current_block}")

                self.do_step(self.current_block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
                    bt.logging.error(f"Error during validation step {self.step}", exc_info=e)
                    continue

                raise


def main():
    Validator().run()


if __name__ == '__main__':
    main()
