from typing import cast

from aiohttp import ClientSession, MultipartReader
from aiohttp.multipart import MultipartResponseWrapper, BodyPartReader
from diffusers import LatentConsistencyModelPipeline
from torch import zeros_like

from neuron.neuron.config import get_config
from neuron.neuron.neuron import Neuron
from neuron.neuron.pipeline import load_pipeline


class Validator(Neuron):
    def __init__(self):
        super().__init__(get_config(type(self)))

        self.pipeline, _ = load_pipeline(self.device)

        self.session = ClientSession()
        self.scores = zeros_like(self.metagraph.n)

        self.step = 0

    async def get_checkpoint(self, uid: int) -> tuple[bytes, float]:
        axon = self.metagraph.axons[uid]

        url = f"http://{axon.ip}:{axon.port}/checkpoint"

        async with self.session.get(url) as response:
            response.raise_for_status()

            reader = MultipartReader.from_response(response)

            average_speed: float | None = None
            checkpoint: bytes | None = None

            async for part in reader:
                part = cast(BodyPartReader, part)

                if part.name == "checkpoint":
                    checkpoint = await part.read()

                    if average_speed:
                        return checkpoint, average_speed
                elif part.name == "average_speed":
                    average_speed = await part.json()

                    if checkpoint:
                        return checkpoint, average_speed
                else:
                    raise RuntimeError(f"Invalid response from miner {uid}")

            raise RuntimeError(f"Invalid response from miner {uid}")

    async def run(self):
        uid = 0

        checkpoint_bytes, average_speed = await self.get_checkpoint(uid)

        file_name = "next_checkpoint_to_check.safetensors"
        with open(file_name, "wb") as checkpoint_file:
            checkpoint_file.write(checkpoint_bytes)

        checkpoint = LatentConsistencyModelPipeline.from_single_file(file_name)

        chec
