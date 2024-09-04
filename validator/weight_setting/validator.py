import time
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date, datetime
from os import makedirs
from os.path import isfile, expanduser, join
from typing import cast, NewType, TypeAlias
from zoneinfo import ZoneInfo

import bittensor as bt
import numpy
import wandb
from bittensor.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from numpy import real, isreal
from numpy.polynomial import Polynomial
from pickle import dump, load
from wandb.sdk.wandb_run import Run

from neuron import (
    CheckpointSubmission,
    get_config,
    ContestId,
    get_submission,
    CURRENT_CONTEST,
    find_contest,
    ContestDeviceValidationError,
)
from base_validator.metrics import Metrics
from wandb_args import add_wandb_args

WEIGHTS_VERSION = 13
VALIDATOR_VERSION = "2.0.0"

WINNER_PERCENTAGE = 0.8
IMPROVEMENT_BENCHMARK_PERCENTAGE = 1.05

Uid = NewType("Uid", int)
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

        self.miner_score_version = state.get("miner_score_version", WEIGHTS_VERSION)
        self.__dict__.update(state)

    def __repr__(self):
        return f"ContestState(id={self.id}, miner_score_version={self.miner_score_version}, miner_info={self.miner_info})"


class Validator:
    config: bt.config
    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
    uid: int

    metrics: Metrics

    hotkeys: list[str]
    step: int

    last_day: date | None
    contest_state: ContestState | None
    previous_day_winners: WinnerList
    should_set_weights: bool

    wandb_run: Run | None

    current_block: int
    last_block_fetch: datetime | None = None

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        from diagnostics import save_validator_diagnostics
        save_validator_diagnostics(self.config)

        bt.logging.info("Setting up bittensor objects")

        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = bt.wallet(config=self.config)

        self.metrics = Metrics(self.metagraph)

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
        self.should_set_weights = False

        self.wandb_run = None

        self.load_state()

        self.contest = find_contest(self.contest_state.id) if self.contest_state else CURRENT_CONTEST

        self.contest.validate()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        day = self.last_day
        hotkey = self.wallet.hotkey.ss58_address
        uid = self.metagraph.hotkeys.index(hotkey)
        netuid = self.metagraph.netuid

        name = f"validator-{uid}-{day.year}-{day.month}-{day.day}"

        signing_message = f"{uid}:{hotkey}:{self.contest_state.id.name}"
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
                "contest": self.contest_state.id.name,
                "signature": signature,
            },
            allow_val_change=True,
            anonymous="allow",
            tags=[
                f"version_{VALIDATOR_VERSION}",
                f"sn{netuid}",
            ],
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def start_wandb_run(self):
        if self.config.wandb.off:
            return

        if self.wandb_run:
            bt.logging.info("New contest day, starting a new wandb run.")

            self.wandb_run.finish()

        self.new_wandb_run()

    @classmethod
    def add_extra_args(cls, argument_parser: ArgumentParser):
        argument_parser.add_argument(
            "--epoch_length",
            type=int,
            help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
            default=100,
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

        return join(full_path, "state.pt")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        with open(self.state_path(), "wb") as file:
            dump(
                {
                    "step": self.step,
                    "hotkeys": self.hotkeys,
                    "metrics": self.metrics,
                    "last_day": self.last_day,
                    "contest_state": self.contest_state,
                    "previous_day_winners": self.previous_day_winners,
                    "should_set_weights": self.should_set_weights,
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
        self.metrics = state.get("metrics", self.metrics)
        self.metrics.set_metagraph(self.metagraph)
        self.last_day = state["last_day"]
        self.contest_state = state["contest_state"]
        self.previous_day_winners = (
            state.get("previous_day_winners", self.previous_day_winners) or
            self.previous_day_winners
        )
        self.should_set_weights = state["should_set_weights"]

        if self.contest_state:
            # Remove outdated checks
            self.contest_state.miner_score_versions = {
                uid: version
                for uid, version in self.contest_state.miner_score_versions.items()
                if version == WEIGHTS_VERSION
            }

            if self.last_day:
                self.start_wandb_run()

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

        self.set_weights()

    def set_weights(self):
        if len(self.hotkeys) != len(self.metagraph.hotkeys):
            self.metrics.resize()

            if self.contest_state:
                new_miner_info = [None] * self.metagraph.n.item()
                length = len(self.hotkeys)
                new_miner_info[:length] = self.contest_state.miner_info[:length]

                self.contest_state.miner_info = new_miner_info

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.metrics.reset(uid)

                filtered_winners = [
                    (winner_uid, score)
                    for winner_uid, score in self.previous_day_winners
                    if uid != winner_uid
                ]

                self.previous_day_winners = filtered_winners

                if self.contest_state:
                    if uid in self.contest_state.miner_score_versions:
                        del self.contest_state.miner_score_versions[uid]

                    self.contest_state.miner_info[uid] = None

        self.hotkeys = self.metagraph.hotkeys

        if not self.should_set_weights:
            bt.logging.info("Will not set weights as contest is not done")

            return

        bt.logging.info("Setting weights")

        buckets = [ContestSubmissionsBucket(scores) for scores in self.get_score_buckets()]

        if len(self.previous_day_winners):
            winners = self.current_winners()

            if len(winners):
                highest_score = max([score for _, score in winners])

                winner_overrides = [
                    (uid, score)
                    for uid, score in self.previous_day_winners if
                    score > highest_score * IMPROVEMENT_BENCHMARK_PERCENTAGE
                ]

                if len(winner_overrides):
                    buckets.append(ContestSubmissionsBucket(winner_overrides, previous_day_winners=True))

        highest_bucket = len(buckets) - 1

        if self.wandb_run:
            log_data = {}

            for index, bucket in enumerate(buckets):
                bucket_rank = highest_bucket - index

                for uid, score in bucket.scores:
                    metric_data = self.metrics.benchmarks[uid]
                    if metric_data:
                        log_data[str(uid)] = {
                            "rank": bucket_rank,
                            "model": cast(CheckpointSubmission, self.contest_state.miner_info[uid]).repository,
                            "baseline_generation_time": metric_data.baseline_average,
                            "generation_time": metric_data.generation_time,
                            "similarity": metric_data.similarity_score,
                            "size": metric_data.size,
                            "baseline_vram_used": metric_data.vram_used,
                            "vram_used": metric_data.vram_used,
                            "baseline_watts_used": metric_data.watts_used,
                            "watts_used": metric_data.watts_used,
                            "hotkey": self.hotkeys[uid],
                            "multiday_winner": bucket.previous_day_winners,
                        }

            self.wandb_run.log(data=log_data)

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

    def get_score_buckets(self) -> list[WinnerList]:
        sorted_contestants = self.metrics.get_sorted_contestants()

        buckets: list[WinnerList] = [[]]

        last_score = sorted_contestants[0][1] if len(sorted_contestants) else None

        for contestant in sorted_contestants:
            _, score = contestant

            if last_score and score > last_score * IMPROVEMENT_BENCHMARK_PERCENTAGE:
                # New bucket
                buckets.append([contestant])
            else:
                buckets[len(buckets) - 1].append(contestant)

            last_score = score

        return buckets

    def current_winners(self) -> WinnerList:
        return [(uid, score) for uid, score in self.get_score_buckets()[-1] if score > 0.0]

    def get_miner_submissions(self):
        visited_repositories: dict[str, tuple[int, int]] = {}

        miner_info: list[CheckpointSubmission | None] = []

        for uid in range(self.metagraph.n.item()):
            submission = get_submission(self.subtensor, self.metagraph, self.metagraph.hotkeys[uid], self.current_block)

            if not submission:
                miner_info.append(None)
                continue

            info, block = submission

            existing_submission = visited_repositories.get(info.image)

            if existing_submission:
                existing_uid, existing_block = existing_submission

                if block > existing_block:
                    miner_info.append(None)
                    continue

                miner_info[existing_uid] = None

            miner_info.append(info)
            visited_repositories[info.image] = uid, block

        return miner_info

    def do_step(self, block: int):
        now = datetime.now(tz=ZoneInfo("America/New_York"))

        if (not self.last_day or self.last_day < now.date()) and now.hour >= 12:
            # Past noon, should start collecting submissions
            self.last_day = now.date()

            bt.logging.info("Collecting all submissions")

            miner_info = self.get_miner_submissions()

            bt.logging.info(f"Got {miner_info} submissions")

            self.should_set_weights = False

            if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
                # New contest, restart
                self.contest = CURRENT_CONTEST

                bt.logging.info(f"Working on contest {self.contest.id.name} today's submissions")

                self.contest.validate()

                self.metrics.clear()

                self.contest_state = ContestState(self.contest.id, miner_info)
                self.previous_day_winners = []
                self.start_wandb_run()
            else:
                def should_update(old_info: CheckpointSubmission | None, new_info: CheckpointSubmission | None):
                    if old_info is None and new_info is None:
                        return False

                    if (old_info is None) != (new_info is None):
                        return True

                    return old_info.repository != new_info.repository or old_info.revision != new_info.revision

                self.start_wandb_run()

                updated_uids = set([
                    uid
                    for uid in range(self.metagraph.n.item())
                    if should_update(self.contest_state.miner_info[uid], miner_info[uid])
                ])

                bt.logging.info(f"Miners {updated_uids} changed their submissions")

                for uid in updated_uids:
                    self.metrics.reset(uid)

                    if uid in self.contest_state.miner_score_versions:
                        del self.contest_state.miner_score_versions[uid]

                self.contest_state.miner_info = miner_info

                winners = self.current_winners()

                if len(winners):
                    if len(self.previous_day_winners):
                        bucket_score = min([score for _, score in self.previous_day_winners])

                        new_winners = [(uid, score) for uid, score in winners if score > bucket_score * IMPROVEMENT_BENCHMARK_PERCENTAGE]

                        if len(new_winners):
                            # New winner
                            self.previous_day_winners = new_winners
                    else:
                        self.previous_day_winners = winners

            self.step += 1
            return

        last_update = self.metagraph.last_update[self.uid]
        blocks_elapsed = block - last_update

        if blocks_elapsed >= self.config.epoch_length:
            bt.logging.info(f"{blocks_elapsed} blocks since last update, resyncing metagraph")
            self.sync()
        else:
            bt.logging.info(
                f"{blocks_elapsed} since last update, "
                f"{self.config.epoch_length - blocks_elapsed} blocks remaining until metagraph sync"
            )

        if self.should_set_weights:
            self.step += 1

            bt.logging.info(f"Nothing to do in this step, sleeping for {self.config.epoch_length} blocks")
            time.sleep(self.config.epoch_length * 12)

            return

        self.test_next_miner()

        self.step += 1

        self.save_state()

    def run(self):
        while True:
            if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
                self.current_block = self.subtensor.get_current_block()
                self.last_block_fetch = datetime.now()

            try:
                bt.logging.info(f"Step {self.step}, block {self.current_block}")

                self.do_step(self.current_block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
                    bt.logging.error(f"Error during validation step {self.step}, {traceback.format_exception(e)}")
                    continue

                raise


if __name__ == '__main__':
    Validator().run()
