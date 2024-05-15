from argparse import ArgumentParser
from asyncio import run
from logging import getLogger, INFO, WARNING, basicConfig, DEBUG
from random import choices, choice

import bittensor as bt
from aiohttp import ClientSession
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
from torch import zeros_like, float32, int64, Tensor

from neuron import (
    AVERAGE_TIME,
    BASELINE_CHECKPOINT,
    CheckpointInfo,
    compare_checkpoints,
    get_checkpoint_info,
    get_config,
    SPEC_VERSION,
    from_pretrained,
    MLPACKAGES,
)

logger = getLogger(__name__)

WINNER_PERCENTAGE = 0.8


def _get_cut(rank: int):
    return WINNER_PERCENTAGE * pow(1 / (1 - WINNER_PERCENTAGE), -rank)


class Validator:
    config: bt.config
    subtensor: bt.subtensor
    metagraph: bt.metagraph
    wallet: bt.wallet
    device: str
    pipeline: CoreMLStableDiffusionPipeline
    session: ClientSession
    scores: Tensor
    miner_last_checked: Tensor
    miner_info: dict[int, CheckpointInfo]
    hotkeys: list[str]
    uid: int
    step: int

    def __init__(self):
        self.config = get_config(Validator.add_extra_args)

        if self.config.logging and self.config.logging.debug:
            basicConfig(level=DEBUG)
        else:
            basicConfig(level=INFO)

        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.wallet = bt.wallet(config=self.config)

        self.pipeline = from_pretrained(BASELINE_CHECKPOINT, MLPACKAGES, self.config.device).coreml_sdxl_pipeline

        self.session = ClientSession()
        self.scores = zeros_like(self.metagraph.S, dtype=float32)
        self.miner_last_checked = zeros_like(self.metagraph.S, dtype=int64)

        self.hotkeys = self.metagraph.hotkeys

        self.uid = self.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.step = 0

    @classmethod
    def add_extra_args(cls, argument_parser: ArgumentParser):
        argument_parser.add_argument(
            "--epoch_length",
            type=int,
            help="The default epoch length (how often we pull the metagraph, measured in 12 second blocks).",
            default=100,
        )

    def get_next_uid(self) -> int | None:
        miner_uids = [
            uid
            for uid in range(self.metagraph.n.item())
            if self.metagraph.axons[uid].is_serving
        ]

        if not len(miner_uids):
            return None

        blocks = [self.miner_last_checked[uid].item() for uid in miner_uids]

        if sum(blocks) == 0:
            return choice(miner_uids)

        last_block = max(blocks)
        weights = [last_block - block for block in blocks]

        return choices(miner_uids, weights=weights)[0]

    def sync(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )

            exit()

        self.metagraph.sync(subtensor=self.subtensor)

        self.set_weights()

    def set_weights(self):
        if len(self.hotkeys) != len(self.metagraph.hotkeys):
            # resize
            new_scores = zeros_like(self.metagraph.S, dtype=float32)
            new_miner_last_checked = zeros_like(self.metagraph.S, dtype=int64)

            length = len(self.hotkeys)
            new_scores[:length] = self.scores[:length]
            new_miner_last_checked[:length] = self.miner_last_checked[:length]

            self.scores = new_scores
            self.miner_last_checked = new_miner_last_checked

        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                # hotkey has been replaced
                self.scores[uid] = 0.0
                self.miner_last_checked[uid] = 0

        sorted_scores = sorted(enumerate(self.scores.tolist()), key=lambda score: score[1], reverse=True)
        ranked_scores = [(index, _get_cut(index)) for index, score in sorted_scores if score > 0.0]

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

        logger.log(level, f"set_weights {state}, {message}")

    async def do_step(self):
        uid = self.get_next_uid()
        axon = self.metagraph.axons[uid]

        try:
            logger.info(f"Checking miner {uid}, hotkey: {axon.hotkey}")

            checkpoint_info = get_checkpoint_info(self.subtensor, self.metagraph, axon.hotkey)

            logger.info(
                f"Miner {uid} returned {checkpoint_info.repository} as the model, "
                f"with a reported speed of {checkpoint_info.average_time}"
            )

            checkpoint = from_pretrained(checkpoint_info.repository, checkpoint_info.mlpackages, self.config.device).coreml_sdxl_pipeline

            comparison = compare_checkpoints(
                self.pipeline,
                checkpoint,
                checkpoint_info.average_time,
            )

            if comparison.failed:
                self.scores[uid] = 0.0
            else:
                self.scores[uid] = min(0, AVERAGE_TIME - comparison.average_time) * comparison.average_similarity
        except Exception as e:
            self.scores[uid] = 0.0
            logger.info(f"Failed to query miner {uid}, {str(e)}")
            logger.debug(f"Miner {uid} error", exc_info=e)

        block = self.subtensor.get_current_block()

        self.miner_last_checked[uid] = block

        if block - self.metagraph.last_update[self.uid] >= self.config.epoch_length:
            self.sync()

        self.step += 1

    async def run(self):
        while True:
            try:
                await self.do_step()
            except Exception as e:
                logger.error(f"Error during validation step {self.step}", exc_info=e)


if __name__ == '__main__':
    run(Validator().run())
