from aiohttp import ClientSession, MultipartReader
from diffusers import LatentConsistencyModelPipeline
from torch import zeros_like

from neuron import get_config, CheckpointInfo, Neuron, download_pipeline

from . import compare_checkpoints


class Validator(Neuron):
    def __init__(self):
        super().__init__(get_config(type(self)))

        self.pipeline = LatentConsistencyModelPipeline.from_pretrained(download_pipeline()).to(self.device)

        self.session = ClientSession()
        self.scores = zeros_like(self.metagraph.n)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.step = 0

    async def get_checkpoint(self, uid: int) -> CheckpointInfo:
        axon = self.metagraph.axons[uid]

        url = f"http://{axon.ip}:{axon.port}/checkpoint"

        async with self.session.get(url) as response:
            response.raise_for_status()

            return CheckpointInfo.parse_obj(await response.json())

    async def run(self):
        uid = 0

        checkpoint_info = await self.get_checkpoint(uid)

        checkpoint = LatentConsistencyModelPipeline.from_pretrained(checkpoint_info.repository)

        self.scores[uid] = compare_checkpoints(self.pipeline, checkpoint, checkpoint_info.average_time)

        if self.subtensor.get_current_block() - self.metagraph.last_update[self.uid] >= self.config.neuron.epoch_length:
            self.sync()

            # TODO set weights
