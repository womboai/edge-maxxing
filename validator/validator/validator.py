import time
import traceback
from argparse import ArgumentParser
from datetime import date, datetime
from os import makedirs
from os.path import isfile, expanduser, join
from random import choice
from zoneinfo import ZoneInfo

import bittensor as bt
import numpy
import wandb
from bittensor.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from numpy import real, isreal
from numpy.polynomial import Polynomial
from torch import save, load
from wandb.sdk.wandb_run import Run

from neuron import (
    CheckpointSubmission,
    compare_checkpoints,
    get_config,
    SPEC_VERSION,
    ContestId,
    get_submission,
    CURRENT_CONTEST,
    find_contest, ContestDeviceValidationError,
)
from .wandb_args import add_wandb_args

WINNER_PERCENTAGE = 0.8
IMPROVEMENT_BENCHMARK_PERCENTAGE = 1.01


def _get_incentive(rank: int, sequence_ratio: float):
    return WINNER_PERCENTAGE * (sequence_ratio ** rank)


def _winner_percentage_sequence_ratio(sample_count: int):
    if sample_count == 1:
        return 1 / WINNER_PERCENTAGE

    polynomial = Polynomial([1 - WINNER_PERCENTAGE, -1] + ([0.0] * (sample_count - 2)) + [WINNER_PERCENTAGE])
    real_roots = [float(real(root)) for root in polynomial.roots() if isreal(root) and root >= 0.0]

    return real_roots[0]


class ContestState:
    id: ContestId
    miners_checked: set[int]
    miner_info: list[CheckpointSubmission | None]

    def __init__(
        self,
        contest_id: ContestId,
        miner_info: list[CheckpointSubmission | None],
    ):
        self.id = contest_id
        self.miners_checked = set()
        self.miner_info = miner_info


class Validator:
    config: bt.config
    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
    uid: int

    scores: list[float]
    sequence_ratio: float = 1.0 - WINNER_PERCENTAGE
    hotkeys: list[str]
    step: int

    last_day: date | None
    contest_state: ContestState | None
    previous_day_winner: tuple[int, float] | None
    should_set_weights: bool

    wandb_run: Run | None

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        bt.logging.info("Setting up bittensor objects")

        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = bt.wallet(config=self.config)

        self.scores = [0.0] * self.metagraph.n.item()

        self.hotkeys = self.metagraph.hotkeys

        self.uid = self.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.step = 0

        self.last_day = None
        self.contest_state = None
        self.previous_day_winner = None
        self.should_set_weights = False

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
            },
            allow_val_change=True,
            anonymous="allow",
            tags=[
                f"version_{SPEC_VERSION}",
                f"sn{netuid}",
            ],
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def start_wandb_run(self):
        if self.config.wandb.off:
            return

        if self.wandb_run:
            bt.logging.info(f"New contest day, starting a new wandb run.")

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
        save(
            {
                "step": self.step,
                "hotkeys": self.hotkeys,
                "scores": self.scores,
                "last_day": self.last_day,
                "contest_state": self.contest_state,
                "winner_override": self.previous_day_winner,
                "should_set_weights": self.should_set_weights,
            },
            self.state_path(),
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        path = self.state_path()

        if not isfile(path):
            return

        # Load the state of the validator from file.
        state = load(path)
        self.step = state["step"]
        self.hotkeys = state["hotkeys"]
        self.scores = state["scores"]
        self.last_day = state["last_day"]
        self.contest_state = state["contest_state"]
        self.previous_day_winner = state.get("winner_override", self.previous_day_winner)
        self.should_set_weights = state["should_set_weights"]

        if self.contest_state:
            self.start_wandb_run()

    def sync(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )

            exit()

        self.metagraph.sync(subtensor=self.subtensor)

        self.set_weights()

    def set_weights(self):
        if len(self.hotkeys) != len(self.metagraph.hotkeys):
            # resize
            new_scores = [0.0] * self.metagraph.n.item()

            length = len(self.hotkeys)
            new_scores[:length] = self.scores[:length]

            self.scores = new_scores

            if self.contest_state:
                new_miner_info = [None] * self.metagraph.n.item()
                new_miner_info[:length] = self.contest_state.miner_info[:length]

                self.contest_state.miner_info = new_miner_info

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.scores[uid] = 0.0

                if self.previous_day_winner and uid == self.previous_day_winner[0]:
                    self.previous_day_winner = None

                if self.contest_state:
                    if uid in self.contest_state.miners_checked:
                        self.contest_state.miners_checked.remove(uid)

                    self.contest_state.miner_info[uid] = None

        self.hotkeys = self.metagraph.hotkeys

        if not self.should_set_weights:
            bt.logging.info("Will not set weights as contest is not done")

            return

        bt.logging.info("Setting weights")

        sorted_uids = [uid for uid, score in sorted(enumerate(self.scores), key=lambda score: score[1], reverse=True)]

        if self.previous_day_winner:
            _, highest_score = self.current_best_contestant()
            winner_uid, winner_score = self.previous_day_winner

            if highest_score >= winner_score * IMPROVEMENT_BENCHMARK_PERCENTAGE:
                sorted_uids = [winner_uid] + [uid for uid in sorted_uids if uid != winner_uid]

        self.wandb_run.log(
            data={
                str(uid): {
                    "rank": rank,
                    "score": self.scores[uid],
                    "multiday_winner": self.previous_day_winner and uid == self.previous_day_winner[0],
                }
                for rank, uid in enumerate(sorted_uids)
            },
            step=self.step,
        )

        ranked_scores = [
            (uid, _get_incentive(index, self.sequence_ratio))
            for index, uid in enumerate(sorted_uids)
        ]

        ranked_scores = sorted(ranked_scores, key=lambda score: score[0])

        uids = numpy.array([uid for uid, _ in ranked_scores])
        weights = numpy.array([weight for _, weight in ranked_scores])

        if numpy.isnan(weights).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0
        raw_weights = weights / numpy.linalg.norm(weights, ord=1, axis=0, keepdims=True)

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", uids)
        # Process the raw weights to final_weights via subtensor limitations.

        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=uids,
            weights=raw_weights,
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
            SPEC_VERSION,
        )

        if result:
            bt.logging.info(f"set_weights successful, {message}")
        else:
            bt.logging.warning(f"set_weights failed, {message}")

    def get_next_uid(self) -> int | None:
        uids = set([uid for uid, info in enumerate(self.contest_state.miner_info) if info])
        remaining_uids = uids - self.contest_state.miners_checked

        if not len(remaining_uids):
            return None

        return choice(list(remaining_uids))

    def test_next_miner(self):
        uid = self.get_next_uid()

        if uid is None:
            if not self.should_set_weights:
                # Finished all submissions
                bt.logging.info(f"Contest {self.contest_state.id} done for {self.last_day}")

                self.should_set_weights = True

            return

        axon = self.metagraph.axons[uid]

        try:
            bt.logging.info(f"Checking miner {uid}, hotkey: {axon.hotkey}")

            checkpoint_info = self.contest_state.miner_info[uid]

            bt.logging.info(
                f"Miner {uid} returned {checkpoint_info.repository} as the model, "
                f"with a reported speed of {checkpoint_info.average_time}"
            )

            comparison = compare_checkpoints(
                self.contest,
                checkpoint_info.repository,
                checkpoint_info.average_time,
            )

            if comparison.failed:
                self.scores[uid] = 0.0
            else:
                self.scores[uid] = min(
                    0.0,
                    comparison.baseline_average - comparison.average_time,
                ) * comparison.average_similarity
        except Exception as e:
            self.scores[uid] = 0.0
            bt.logging.info(f"Failed to query miner {uid}, {e}")
            bt.logging.debug(f"Miner {uid} error, {traceback.format_exception(e)}")

        self.contest_state.miners_checked.add(uid)

    def current_best_contestant(self) -> tuple[int, float]:
        return max(enumerate(self.scores), key=lambda contestant: contestant[1])

    def get_miner_submissions(self):
        visited_repositories: dict[str, tuple[int, int]] = {}

        miner_info: list[CheckpointSubmission | None] = []

        for uid in range(self.metagraph.n.item()):
            submission = get_submission(self.subtensor, self.metagraph, self.metagraph.hotkeys[uid])

            if not submission:
                miner_info.append(None)
                continue

            info, block = submission

            existing_submission = visited_repositories.get(info.repository)

            if existing_submission:
                existing_uid, existing_block = existing_submission

                if block > existing_block:
                    miner_info.append(None)
                    continue

                miner_info[existing_uid] = None

            miner_info.append(info)
            visited_repositories[info.repository] = uid, block

        return miner_info

    def do_step(self, block: int):
        now = datetime.now(tz=ZoneInfo("America/New_York"))

        if (not self.last_day or self.last_day < now.date()) and now.hour >= 12:
            # Past noon, should start collecting submissions
            self.last_day = now.date()

            self.start_wandb_run()

            bt.logging.info("Collecting all submissions")

            miner_info = self.get_miner_submissions()

            self.should_set_weights = False

            if not self.contest_state or self.contest_state.id != CURRENT_CONTEST.id:
                # New contest, restart
                self.contest = CURRENT_CONTEST

                bt.logging.info(f"Working on contest {self.contest.id.name} today's submissions")

                self.contest.validate()

                self.scores = [0.0] * self.metagraph.n.item()

                self.contest_state = ContestState(self.contest.id, miner_info)
                self.previous_day_winner = None

                bt.logging.info(f"Got the following valid submissions: {list(enumerate(miner_info))}")
            else:
                def should_update(old_info: CheckpointSubmission | None, new_info: CheckpointSubmission | None):
                    if not old_info and not new_info:
                        return False

                    if (not old_info) != (not new_info):
                        return True

                    return old_info.repository != new_info.repository

                updated_uids = set([
                    uid
                    for uid in range(self.metagraph.n.item())
                    if should_update(self.contest_state.miner_info[uid], miner_info[uid])
                ])

                for uid in updated_uids:
                    self.scores[uid] = 0.0

                self.contest_state.miner_info = miner_info
                self.contest_state.miners_checked -= updated_uids

                highest_uid, highest_score = self.current_best_contestant()

                if self.previous_day_winner:
                    winner_score = self.previous_day_winner[1]

                    if highest_score >= winner_score * IMPROVEMENT_BENCHMARK_PERCENTAGE:
                        # New winner
                        self.previous_day_winner = highest_uid, highest_score
                else:
                    self.previous_day_winner = highest_uid, highest_score

                bt.logging.info(f"Miners {updated_uids} changed their submissions")

            self.sequence_ratio = _winner_percentage_sequence_ratio(len(miner_info) - miner_info.count(None))

            self.step += 1
            return

        last_update = self.metagraph.last_update[self.uid]
        blocks_elapsed = block - last_update

        if blocks_elapsed >= self.config.epoch_length:
            bt.logging.info(f"{blocks_elapsed} since last update, resyncing metagraph and setting weights")
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
            block = self.subtensor.get_current_block()

            try:
                bt.logging.info(f"Step {self.step}, block {block}")

                self.do_step(block)
            except Exception as e:
                if not isinstance(e, ContestDeviceValidationError):
                    bt.logging.error(f"Error during validation step {self.step}, {traceback.format_exception(e)}")
                    continue

                raise


if __name__ == '__main__':
    Validator().run()
