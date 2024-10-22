from operator import itemgetter

from neuron.submission_tester import CheckpointBenchmark, MetricData

TIER_SCORE_IMPROVEMENT_THRESHOLD = 1.05


def get_contestant_scores(benchmarks: list[CheckpointBenchmark | None], baseline_metrics: MetricData):
    contestants = [
        (uid, metric_data.calculate_score(baseline_metrics))
        for uid, metric_data in enumerate(benchmarks)
        if metric_data
    ]

    sorted_contestants = sorted(contestants, key=itemgetter(1))

    return sorted_contestants


def get_scores(contestants: list[tuple[int, float]], node_count: int) -> list[float]:
    if not contestants:
        return []

    last_uid, last_tier_score = contestants[0]

    scores = [0.0] * node_count
    winning_uids = []

    uid = last_uid

    for contestant in contestants:
        uid, score = contestant

        if score > last_tier_score * TIER_SCORE_IMPROVEMENT_THRESHOLD:
            # No longer in top threshold
            last_tier_score = score
            winning_uids.append(last_uid)

        last_uid = uid

    winning_uids.append(uid)

    for winner_index, uid in enumerate(reversed(winning_uids)):
        scores[uid] = 0.5 * (0.5 ** winner_index)

    return scores
