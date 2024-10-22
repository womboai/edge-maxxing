from dataclasses import dataclass
from collections.abc import Iterator

from imagehash import ImageHash

from neuron.submission_tester import GENERATION_TIME_DIFFERENCE_THRESHOLD


@dataclass
class PotentiallyDuplicateSubmissionInfo:
    image_hash: ImageHash
    generation_time: float
    block: int


def find_duplicates(benchmark_info: list[PotentiallyDuplicateSubmissionInfo | None]) -> Iterator[tuple[int, int]]:
    duplicate_buckets: list[set[int]] = []

    for uid_a, benchmark_a in enumerate(benchmark_info):
        if not benchmark_a:
            continue

        for uid_b, benchmark_b in enumerate(benchmark_info):
            if not benchmark_b:
                continue

            if uid_a == uid_b:
                continue

            if (
                not (benchmark_b.image_hash - benchmark_a.image_hash)
                and abs(benchmark_b.generation_time - benchmark_a.generation_time) < GENERATION_TIME_DIFFERENCE_THRESHOLD
            ):
                matching_buckets = [bucket for bucket in duplicate_buckets if uid_a in bucket or uid_b in bucket]
                if len(matching_buckets):
                    bucket = matching_buckets[0]
                    bucket.add(uid_a)
                    bucket.add(uid_b)
                else:
                    duplicate_buckets.append({uid_a, uid_b})

    for bucket in duplicate_buckets:
        oldest = min(bucket, key=lambda uid: benchmark_info[uid].block)

        for uid in bucket:
            if uid != oldest:
                yield uid, oldest
