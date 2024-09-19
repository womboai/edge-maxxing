from typing import TypeAlias

from pydantic import BaseModel

from .contest import ContestId, CURRENT_CONTEST
from .network_commitments import Encoder, Decoder

Uid: TypeAlias = int
Key: TypeAlias = str

SPEC_VERSION = 4


class CheckpointSubmission(BaseModel):
    repository: str
    revision: str
    contest: ContestId = CURRENT_CONTEST.id

    def encode(self, encoder: Encoder):
        encoder.write_str(self.repository)
        encoder.write_str(self.revision)
        encoder.write_uint16(self.contest.value)

    @classmethod
    def decode(cls, decoder: Decoder):
        repository = decoder.read_str()
        revision = decoder.read_str()
        contest_id = ContestId(decoder.read_uint16())

        return cls(
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
