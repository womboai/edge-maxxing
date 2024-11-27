from base.checkpoint import Key, Submissions, Benchmarks
from base.contest import Metrics

TIER_SCORE_IMPROVEMENT_THRESHOLD = 1.05
WINNER_PERCENTAGE = 0.80

def get_contestant_scores(
    submissions: Submissions,
    benchmarks: Benchmarks,
    baseline: Metrics,
) -> dict[Key, float]:
    return {
        key: submissions[key].contest().calculate_score(baseline, benchmark)
        for key, benchmark in benchmarks.items()
    }

def get_contestant_tiers(
    submitted_blocks: dict[Key, int],
    scores: dict[Key, float],
) -> dict[Key, int]:
    return {}  # TODO

def calculate_weights(
    node_count: int,
    tiers: dict[Key, int],
) -> dict[Key, float]:
    return {}  # TODO
