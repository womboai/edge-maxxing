from typing import TypeAlias, Annotated

from pydantic import BaseModel, Field

from .contest import ContestId, CURRENT_CONTEST
from .network_commitments import Encoder, Decoder

Uid: TypeAlias = int
Key: TypeAlias = str

SPEC_VERSION = 5
REVISION_LENGTH = 7


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


def should_update(old_info: CheckpointSubmission | None, new_info: CheckpointSubmission | None):
    if old_info is None and new_info is None:
        return False

    if (old_info is None) != (new_info is None):
        return True

    return old_info.repository != new_info.repository or old_info.revision != new_info.revision
