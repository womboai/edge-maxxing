from operator import itemgetter

from neuron import Uid
from neuron.submission_tester import CheckpointBenchmark, MetricData

TIER_SCORE_IMPROVEMENT_THRESHOLD = 1.05
WINNER_PERCENTAGE = 0.5


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
            # New tier
            last_tier_score = score
            tiers.append([])

        tiers[-1].append(uid)

    return tiers


def get_scores(tiers: list[list[Uid]], blocks: list[int | None], node_count: int) -> list[float]:
    if not tiers:
        return [1.0] * node_count

    ordered_tiers = [
        sorted(tier, key=blocks.__getitem__) for tier in tiers
    ]

    modified_tiers = []

    last_tier = None

    for tier in reversed(ordered_tiers):
        if last_tier:
            modified_tiers.append([tier[0], *last_tier[1:]])
        else:
            modified_tiers.append([tier[0]])

        last_tier = tier

    modified_tiers.append(last_tier[1:])

    scores = [0.0] * node_count

    for index, tier in enumerate(modified_tiers):
        incentive_pool = WINNER_PERCENTAGE * ((1 - WINNER_PERCENTAGE) ** index)
        score = incentive_pool / len(tier)

        for uid in tier:
            scores[uid] = score

    return scores
