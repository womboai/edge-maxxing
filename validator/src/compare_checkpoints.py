from os import urandom

from diffusers import LatentConsistencyModelPipeline


def compare_checkpoints(baseline: LatentConsistencyModelPipeline, miner_checkpoint: LatentConsistencyModelPipeline):
    seed = int.from_bytes(urandom(4), "little")


