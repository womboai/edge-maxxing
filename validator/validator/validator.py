import time
import traceback
from argparse import ArgumentParser
from datetime import date, datetime
from logging import INFO, WARNING, basicConfig, DEBUG
from os import makedirs
from os.path import isfile, expanduser, join
from random import choice
from zoneinfo import ZoneInfo

import bittensor as bt
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
from torch import save, load

from neuron import (
    BASELINE_CHECKPOINT,
    CheckpointSubmission,
    compare_checkpoints,
    get_config,
    SPEC_VERSION,
    from_pretrained,
    ContestId,
    CURRENT_CONTEST,
    get_submission,
)

WINNER_PERCENTAGE = 0.8


def _get_cut(rank: int):
    return WINNER_PERCENTAGE * pow(1 - WINNER_PERCENTAGE, rank)


class Validator:
    config: bt.config
    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
    pipeline: CoreMLStableDiffusionPipeline
    uid: int

    scores: list[float]
    miner_info: list[CheckpointSubmission | None]
    hotkeys: list[str]
    step: int

    last_day: date | None
    working_on: ContestId | None
    miners_checked: set[int]
    should_set_weights: bool

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        bt.logging.info("Setting up bittensor objects")

        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = bt.wallet(config=self.config)

        bt.logging.info("Loading pipeline")

        self.pipeline = from_pretrained(BASELINE_CHECKPOINT, self.config.device).coreml_sdxl_pipeline

        self.scores = [0.0] * self.metagraph.n.item()

        self.hotkeys = self.metagraph.hotkeys

        self.uid = self.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.step = 0

        self.miner_info = [None] * self.metagraph.n.item()
        self.last_day = None
        self.working_on = None
        self.miners_checked = set()
        self.should_set_weights = False

        self.load_state()

    @classmethod
    def add_extra_args(cls, argument_parser: ArgumentParser):
        argument_parser.add_argument(
            "--epoch_length",
            type=int,
            help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
            default=100,
        )

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
                "working_on": self.working_on,
                "miners_checked": self.miners_checked,
                "should_set_weights": self.should_set_weights,
                "miner_info": self.miner_info,
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
        self.working_on = state["working_on"]
        self.miners_checked = state["miners_checked"]
        self.should_set_weights = state["should_set_weights"]
        self.miner_info = state["miner_info"]

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
            new_miner_info = [None] * self.metagraph.n.item()

            length = len(self.hotkeys)
            new_scores[:length] = self.scores[:length]
            new_miner_info[:length] = self.miner_info[:length]

            self.scores = new_scores
            self.miner_info = new_miner_info

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.scores[uid] = 0.0
                self.miners_checked.remove(uid)
                self.miner_info[uid] = None

        if not self.should_set_weights:
            return

        sorted_scores = sorted(enumerate(self.scores), key=lambda score: score[1], reverse=True)
        ranked_scores = [(index, _get_cut(index) if score > 0.0 else 0.0) for index, score in sorted_scores]

        weights = sorted(ranked_scores, key=lambda score: score[0])

        uids = [uid for uid, _ in weights]
        weights = [weights for _, weights in weights]

        result, message = self.subtensor.set_weights(
            self.wallet,
            self.metagraph.netuid,
            uids,
            weights,
            SPEC_VERSION,
        )

        state, level = ("successful", INFO) if result else ("failed", WARNING)

        bt.logging.log(level, f"set_weights {state}, {message}")
        self.should_set_weights = False

    def get_next_uid(self) -> int | None:
        uids = set([uid for uid, info in enumerate(self.miner_info) if info])
        remaining_uids = uids - self.miners_checked

        if not len(remaining_uids):
            return None

        return choice(list(remaining_uids))

    def test_next_miner(self):
        uid = self.get_next_uid()

        if uid is None:
            # Finished all submissions
            self.working_on = None
            return

        axon = self.metagraph.axons[uid]

        try:
            bt.logging.info(f"Checking miner {uid}, hotkey: {axon.hotkey}")

            checkpoint_info = self.miner_info[uid]

            bt.logging.info(
                f"Miner {uid} returned {checkpoint_info.repository} as the model, "
                f"with a reported speed of {checkpoint_info.average_time}"
            )

            checkpoint = from_pretrained(checkpoint_info.repository, self.config.device).coreml_sdxl_pipeline

            comparison = compare_checkpoints(
                self.pipeline,
                checkpoint,
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
            bt.logging.info(f"Failed to query miner {uid}", suffix=e)
            bt.logging.debug(f"Miner {uid} error", suffix=traceback.format_exception(e))

        self.miners_checked.add(uid)
        self.should_set_weights = True

    def do_step(self, block: int):
        now = datetime.now(tz=ZoneInfo("America/New_York"))

        if not self.working_on and (not self.last_day or self.last_day < now.date()) and now.hour >= 11:
            # Past noon, should start collecting submissions
            bt.logging.info(f"Working on contest {CURRENT_CONTEST} today's submission")

            self.last_day = now.date()
            self.working_on = CURRENT_CONTEST
            self.miners_checked = set()

            bt.logging.info("Collecting all submissions")
            self.miner_info = [
                get_submission(self.subtensor, self.metagraph, self.metagraph.hotkeys[uid])
                for uid in range(self.metagraph.n.item())
            ]

            bt.logging.info(f"Got the following valid submissions: {list(enumerate(self.miner_info))}")

            self.step += 1

            return

        if block - self.metagraph.last_update[self.uid] >= self.config.epoch_length:
            self.sync()

        if not self.working_on:
            self.step += 1

            bt.logging.info(f"Nothing to do in this step, sleeping for {self.config.epoch_length} blocks")
            time.sleep(self.config.epoch_length * 12)

            return

        self.test_next_miner()

        self.step += 1

        self.save_state()

    def run(self):
        while True:
            try:
                block = self.subtensor.get_current_block()
                bt.logging.info(f"Step {self.step}, block {block}")

                self.do_step(block)
            except Exception as e:
                bt.logging.error(f"Error during validation step {self.step}", suffix=traceback.format_exception(e))


if __name__ == '__main__':
    Validator().run()
