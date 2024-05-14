from typing import TypeAlias

import bittensor as bt
from diffusers import LatentConsistencyModelPipeline
from pydantic import BaseModel

BASELINE_CHECKPOINT = "SimianLuo/LCM_Dreamshaper_v7"
AVERAGE_TIME = 3.0
SPEC_VERSION = 20


PipelineType: TypeAlias = LatentConsistencyModelPipeline


class CheckpointInfo(BaseModel):
    repository: str = BASELINE_CHECKPOINT
    average_time: float = AVERAGE_TIME
    spec_version: int = SPEC_VERSION


def get_checkpoint_info(subtensor: bt.subtensor, metagraph: bt.metagraph, uid: int) -> CheckpointInfo | None:
    return CheckpointInfo.model_validate_json(subtensor.get_commitment(metagraph.netuid, uid))
