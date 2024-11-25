from fiber.chain.commitments import publish_raw_commitment, _deserialize_commitment_field
from fiber.logging_utils import get_logger
from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.storage import StorageKey

from . import ACTIVE_CONTESTS
from .checkpoint import CheckpointSubmission, SPEC_VERSION, Key, MinerModelInfo
from .contest import ModelRepositoryInfo, find_contest
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


def get_submissions(
    substrate: SubstrateInterface,
    hotkeys: list[Key],
    netuid: int,
    block: int
) -> list[MinerModelInfo | None]:
    submissions: list[MinerModelInfo | None] = [None] * len(hotkeys)

    storage_keys: list[StorageKey] = []
    for hotkey in hotkeys:
        storage_keys.append(substrate.create_storage_key(
            "Commitments",
            "CommitmentOf",
            [netuid, hotkey]
        ))

    commitments = substrate.query_multi(
        storage_keys=storage_keys,
        block_hash=substrate.get_block_hash(block),
    )

    for storage, commitment in commitments:
        hotkey = storage.params[1]
        try:
            if not commitment or not commitment.value:
                continue

            fields = commitment.value["info"]["fields"]
            if not fields:
                continue

            field = _deserialize_commitment_field(fields[0])
            if field is None:
                continue

            decoder = Decoder(field[1])
            spec_version = decoder.read_uint16()
            if spec_version != SPEC_VERSION:
                continue

            while not decoder.eof:
                info = CheckpointSubmission.decode(decoder)
                repository_url = info.get_repo_link()

                if info.contest not in ACTIVE_CONTESTS:
                    continue

                if repository_url == find_contest(info.contest).baseline_repository.url:
                    continue

                repository = ModelRepositoryInfo(url=repository_url, revision=info.revision)
                submitted_block = int(commitment.value["block"])
                submissions[hotkeys.index(hotkey)] = MinerModelInfo(repository, info.contest, submitted_block)
        except Exception as e:
            logger.error(f"Failed to get submission from miner {hotkey}")
            logger.error(f"Submission parsing error", exc_info=e)
            continue

    return submissions