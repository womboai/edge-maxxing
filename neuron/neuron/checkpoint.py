from dataclasses import dataclass
from typing import TypeAlias, Annotated

from pydantic import BaseModel, Field

from .contest import ContestId, CURRENT_CONTEST, ModelRepositoryInfo
from .network_commitments import Encoder, Decoder

Uid: TypeAlias = int
Key: TypeAlias = str

SPEC_VERSION = 5
REVISION_LENGTH = 7


@dataclass
class GenerationOutput:
    prompt: str
    seed: int
    output: bytes
    generation_time: float
    vram_used: float
    watts_used: float


class CheckpointSubmission(BaseModel):
    provider: str
    repository: str
    revision: Annotated[str, Field(min_length=REVISION_LENGTH, max_length=REVISION_LENGTH)]
    contest: ContestId = CURRENT_CONTEST.id

    def encode(self, encoder: Encoder):
        encoder.write_str(self.provider)
        encoder.write_str(self.repository)
        encoder.write_sized_str(self.revision)
        encoder.write_uint16(self.contest.value)

    @classmethod
    def decode(cls, decoder: Decoder):
        provider = decoder.read_str()
        repository = decoder.read_str()
        revision = decoder.read_sized_str(REVISION_LENGTH)
        contest_id = ContestId(decoder.read_uint16())

        return cls(
            provider=provider,
            repository=repository,
            revision=revision,
            contest=contest_id,
        )

    def get_repo_link(self):
        return f"https://{self.provider}/{self.repository}"


class MinerModelInfo:
    repository: ModelRepositoryInfo
    block: int

    def __init__(self, repository: ModelRepositoryInfo, block: int):
        self.repository = repository
        self.block = block
