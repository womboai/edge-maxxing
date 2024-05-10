import logging
from random import choices, choice

from aiohttp import ClientSession
from diffusers import LatentConsistencyModelPipeline
from torch import zeros_like, float32, int64

from neuron import get_config, CheckpointInfo, Neuron, BASELINE_CHECKPOINT
from . import compare_checkpoints

logger = logging.getLogger(__name__)


class Validator(Neuron):
    def __init__(self):
        super().__init__(get_config(type(self)))

        self.pipeline = LatentConsistencyModelPipeline.from_pretrained(BASELINE_CHECKPOINT).to(self.device)

        self.session = ClientSession()
        self.scores = zeros_like(self.metagraph.S, dtype=float32)
        self.miner_last_checked = zeros_like(self.metagraph.S, dtype=int64)

        self.hotkeys = self.metagraph.hotkeys

        self.uid = self.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.step = 0

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
        super().sync()

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

        # TODO set weights

    async def get_checkpoint(self, uid: int) -> CheckpointInfo:
        axon = self.metagraph.axons[uid]

        url = f"http://{axon.ip}:{axon.port}/checkpoint"

        async with self.session.get(url) as response:
            response.raise_for_status()

            return CheckpointInfo.parse_obj(await response.json())

    async def do_step(self):
        uid = self.get_next_uid()
        axon = self.metagraph.axons[uid]

        try:
            logger.info(f"Checking miner {uid}, hotkey: {axon.hotkey}")

            checkpoint_info = await self.get_checkpoint(uid)

            logger.info(
                f"Miner {uid} returned {checkpoint_info.repository} as the model, "
                f"with a reported speed of {checkpoint_info.average_time}"
            )

            checkpoint = LatentConsistencyModelPipeline.from_pretrained(checkpoint_info.repository).to(self.device)
        except Exception as e:
            self.scores[uid] = 0.0
            logger.info(f"Failed to query miner {uid}, {str(e)}")
            logger.debug(f"Miner {uid} error", exc_info=e)
        else:
            self.scores[uid] = compare_checkpoints(self.pipeline, checkpoint, checkpoint_info.average_time)

        block = self.subtensor.get_current_block()

        self.miner_last_checked[uid] = block

        if block - self.metagraph.last_update[self.uid] >= self.config.neuron.epoch_length:
            self.sync()

        self.step += 1

    async def run(self):
        while True:
            try:
                await self.do_step()
            except Exception as e:
                logger.error(f"Error during validation step {self.step}", exc_info=e)
