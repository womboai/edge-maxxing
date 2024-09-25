from fiber.chain.commitments import publish_raw_commitment, get_raw_commitment
from fiber.logging_utils import get_logger
from substrateinterface import SubstrateInterface, Keypair

from .checkpoint import CheckpointSubmission, SPEC_VERSION, Key
from .contest import CURRENT_CONTEST
from .network_commitments import Encoder, Decoder


logger = get_logger(__name__)


def make_submission(
    substrate: SubstrateInterface,
    netuid: int,
    keypair: Keypair,
    submissions: list[CheckpointSubmission],
):
    encoder = Encoder()

    encoder.write_uint16(SPEC_VERSION)

    for submission in submissions:
        submission.encode(encoder)

    data = encoder.finish()

    publish_raw_commitment(
        substrate,
        keypair,
        netuid,
        data,
        wait_for_finalization=False,
    )


def get_submission(
    substrate: SubstrateInterface,
    netuid: int,
    hotkey: Key,
) -> tuple[CheckpointSubmission, int] | None:
    try:
        commitment = get_raw_commitment(substrate, netuid, hotkey)

        if not commitment:
            return None

        decoder = Decoder(commitment.data)

        spec_version = decoder.read_uint16()

        if spec_version != SPEC_VERSION:
            return None

        while not decoder.eof:
            info = CheckpointSubmission.decode(decoder)

            if (
                info.contest != CURRENT_CONTEST.id or
                info.repository == CURRENT_CONTEST.baseline_repository or
                info.revision == CURRENT_CONTEST.baseline_revision
            ):
                continue

            return info, commitment.block

        return None
    except Exception as e:
        logger.error(f"Failed to get submission from miner {hotkey}")
        logger.debug(f"Submission parsing error", exc_info=e)
        return None
