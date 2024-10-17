from dataclasses import dataclass

from imagehash import ImageHash


@dataclass
class PotentiallyDuplicateSubmissionInfo:
    image_hash: ImageHash
    generation_time: float
    block: int


def find_duplicates(benchmark_hashes: list[PotentiallyDuplicateSubmissionInfo | None]):
    duplicate_buckets: list[set[int]] = []

    for uid_a, benchmark_a in enumerate(benchmark_hashes):
        if not benchmark_a:
            continue

        hash_a, _ = benchmark_a

        for uid_b, benchmark_b in enumerate(benchmark_hashes):
            if not benchmark_b:
                continue

            hash_b, _ = benchmark_b

            if uid_a == uid_b:
                continue

            diff = hash_a - hash_b

            if not diff:
                matching_buckets = [bucket for bucket in duplicate_buckets if uid_a in bucket or uid_b in bucket]
                if len(matching_buckets):
                    bucket = matching_buckets[0]
                    bucket.add(uid_a)
                    bucket.add(uid_b)
                else:
                    duplicate_buckets.append({uid_a, uid_b})

    for bucket in duplicate_buckets:
        oldest = min(bucket, key=lambda uid: benchmark_hashes[uid][1])

        for uid in bucket:
            if uid != oldest:
                yield uid
