from operator import itemgetter

from base_validator.metrics import CheckpointBenchmark

from neuron import Uid

WINNERS_SCORE_THRESHOLD = 1.05


def get_contestant_scores(benchmarks: list[CheckpointBenchmark | None]):
    contestants = [
        (uid, metric_data.calculate_score())
        for uid, metric_data in enumerate(benchmarks)
        if metric_data
    ]

    sorted_contestants = sorted(contestants, key=itemgetter(1), reverse=True)

    return sorted_contestants


def get_highest_uids(contestants: list[tuple[int, float]]) -> list[Uid]:
    highest_uids: list[Uid] = []

    if not contestants:
        return []

    _, top_score = contestants[0]

    for contestant in contestants:
        uid, score = contestant

        if top_score > score * WINNERS_SCORE_THRESHOLD:
            # No longer in top threshold
            break
        else:
            highest_uids.append(uid)

    return highest_uids
