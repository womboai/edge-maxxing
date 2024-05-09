import logging
from os.path import basename

from aiohttp import ClientSession, MultipartReader
from diffusers import LatentConsistencyModelPipeline
from torch import zeros_like, float32

from neuron import get_config, CheckpointInfo, Neuron, BASELINE_CHECKPOINT

from . import compare_checkpoints


logger = logging.getLogger(basename(__file__))


class Validator(Neuron):
    def __init__(self):
        super().__init__(get_config(type(self)))

        self.pipeline = LatentConsistencyModelPipeline.from_pretrained(BASELINE_CHECKPOINT).to(self.device)

        self.session = ClientSession()
        self.scores = zeros_like(self.metagraph.S, dtype=float32)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.step = 0

    async def get_checkpoint(self, uid: int) -> CheckpointInfo:
        axon = self.metagraph.axons[uid]

        url = f"http://{axon.ip}:{axon.port}/checkpoint"

        async with self.session.get(url) as response:
            response.raise_for_status()

            return CheckpointInfo.parse_obj(await response.json())

    async def do_step(self):
        uid = 0
        axon = self.metagraph.axons[uid]

        try:
            logger.info(f"Checking miner {uid}, hotkey: {axon.hotkey}")

            checkpoint_info = await self.get_checkpoint(uid)

            logger.info(
                f"Miner {uid} returned {checkpoint_info.repository} as the model, "
                f"with a reported speed of {checkpoint_info.average_time}"
            )

            checkpoint = LatentConsistencyModelPipeline.from_pretrained(checkpoint_info.repository)
        except Exception as e:
            self.scores[uid] = 0.0
            logger.info(f"Failed to query miner {uid}", exc_info=e)
        else:
            self.scores[uid] = compare_checkpoints(self.pipeline, checkpoint, checkpoint_info.average_time)

        if self.subtensor.get_current_block() - self.metagraph.last_update[self.uid] >= self.config.neuron.epoch_length:
            self.sync()

            # TODO set weights

        self.step += 1

    async def run(self):
        while True:
            try:
                await self.do_step()
            except Exception as e:
                logger.error(f"Error during validation step {self.step}", exc_info=e)
