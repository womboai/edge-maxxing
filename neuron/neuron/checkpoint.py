import traceback
from typing import cast, Any

import bittensor as bt
from bittensor.extrinsics.serving import get_metadata, publish_metadata
from pydantic import BaseModel

from .contest import ContestId, CURRENT_CONTEST
from .network_commitments import Encoder, Decoder

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


def make_submission(
    subtensor: bt.subtensor,
    metagraph: bt.metagraph,
    wallet: bt.wallet,
    submission: CheckpointSubmission,
):
    encoder = Encoder()

    encoder.write_uint16(SPEC_VERSION)

    submission.encode(encoder)

    data = encoder.finish()

    publish_metadata(
        subtensor,
        wallet,
        metagraph.netuid,
        f"Raw{len(data)}",
        data,
        wait_for_finalization=False,
    )


def get_submission(
    subtensor: bt.subtensor,
    metagraph: bt.metagraph,
    hotkey: str,
    block: int | None = None
) -> tuple[CheckpointSubmission, int] | None:
    try:
        metadata = cast(dict[str, Any], get_metadata(subtensor, metagraph.netuid, hotkey, block))

        if not metadata:
            return None

        block: int = metadata["block"]
        commitment: dict[str, str] = metadata["info"]["fields"][0]
        hex_data = commitment.values().__iter__().__next__()
        data = bytes.fromhex(hex_data[2:])
        decoder = Decoder(data)

        spec_version = decoder.read_uint16()

        if spec_version != SPEC_VERSION:
            return None

        info = CheckpointSubmission.decode(decoder)

        if (
            info.contest != CURRENT_CONTEST.id or
            info.repository == CURRENT_CONTEST.baseline_repository or
            info.revision == CURRENT_CONTEST.baseline_revision
        ):
            return None

        return info, block
    except Exception as e:
        bt.logging.error(f"Failed to get submission from miner {hotkey}, {e}")
        bt.logging.debug(f"Submission parsing error, {traceback.format_exception(e)}")
        return None
