from operator import itemgetter
from typing import cast

from neuron import Uid
from neuron.submission_tester import CheckpointBenchmark, MetricData

TIER_SCORE_IMPROVEMENT_THRESHOLD = 1.075


def get_contestant_scores(benchmarks: list[CheckpointBenchmark | None], baseline_metrics: MetricData):
    contestants = [
        (uid, metric_data.calculate_score(baseline_metrics))
        for uid, metric_data in enumerate(benchmarks)
        if metric_data
    ]

    sorted_contestants = sorted(contestants, key=itemgetter(1))

    return sorted_contestants


def get_tiers(contestants: list[tuple[Uid, float]]) -> list[list[Uid]]:
    if not contestants:
        return []

    _, last_tier_score = contestants[0]

    tiers = [[]]

    for contestant in contestants:
        uid, score = contestant

        if score > last_tier_score * TIER_SCORE_IMPROVEMENT_THRESHOLD:
            # No longer in top threshold
            last_tier_score = score
            tiers.append([])

        tiers[-1].append(uid)

    return tiers


def pick_winners(tiers: list[list[Uid]], blocks: list[int | None]) -> list[Uid]:
    return [cast(int, min(tier, key=blocks.__getitem__)) for tier in tiers]


def get_scores(winners: list[Uid], node_count: int) -> list[float]:
    scores = [0.0] * node_count

    for index, winner in enumerate(reversed(winners)):
        scores[winner] = 0.5 * (0.5 ** index)

    return scores
