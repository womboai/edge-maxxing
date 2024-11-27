from typing import Annotated

from fiber.chain.commitments import publish_raw_commitment, _deserialize_commitment_field
from fiber.chain.metagraph import Metagraph
from fiber.logging_utils import get_logger
from pydantic import BaseModel, Field
from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.storage import StorageKey

from .checkpoint import SPEC_VERSION, Submissions
from .contest import RepositoryInfo, find_contest, Submission, ContestId, ACTIVE_CONTESTS
from .inputs_api import is_blacklisted, blacklisted_keys
from .network_commitments import Encoder, Decoder

logger = get_logger(__name__)

REVISION_LENGTH = 7

class CheckpointSubmission(BaseModel):
    repository: str
    revision: Annotated[str, Field(min_length=REVISION_LENGTH, max_length=REVISION_LENGTH)]
    contest_id: ContestId

    def encode(self, encoder: Encoder):
        repository = self.repository.replace("https://", "")
        parts = repository.split("/")
        provider = parts[0]
        owner = parts[1]
        repository_name = parts[2]

        encoder.write_str(provider)
        encoder.write_str(f"{owner}/{repository_name}")
        encoder.write_sized_str(self.revision)
        encoder.write_uint16(self.contest_id.value)

    @classmethod
    def decode(cls, decoder: Decoder):
        provider = decoder.read_str()
        repository = decoder.read_str()
        revision = decoder.read_sized_str(REVISION_LENGTH)
        contest_id = ContestId(decoder.read_uint16())

        return cls(
            repository=f"https://{provider}/{repository}",
            revision=revision,
            contest_id=contest_id,
        )

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
    metagraph: Metagraph,
    block: int
) -> Submissions:
    submissions: Submissions = {}
    storage_keys: list[StorageKey] = []

    blacklist = blacklisted_keys()
    hotkeys = [hotkey for hotkey, node in metagraph.nodes.items() if not is_blacklisted(blacklist, hotkey, node.coldkey)]
    for hotkey in hotkeys:
        storage_keys.append(substrate.create_storage_key(
            "Commitments",
            "CommitmentOf",
            [metagraph.netuid, hotkey]
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

                if info.contest_id not in ACTIVE_CONTESTS:
                    continue

                if info.repository == find_contest(info.contest_id).baseline_repository.url:
                    continue

                repository_info = RepositoryInfo(url=info.repository, revision=info.revision)
                submitted_block = int(commitment.value["block"])
                submissions[hotkey] = Submission(
                    repository_info=repository_info,
                    contest_id=info.contest_id,
                    block=submitted_block
                )
        except Exception as e:
            logger.error(f"Failed to get submission from miner {hotkey}: {e}")
            continue

    return submissions
