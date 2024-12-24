from operator import itemgetter

from base.checkpoint import Key, Submissions, Benchmarks
from base.contest import Metrics


def get_contestant_scores(
    submissions: Submissions,
    benchmarks: Benchmarks,
    baseline: Metrics,
) -> dict[Key, float]:
    return {
        key: submissions[key].contest().calculate_score(baseline, benchmark)
        for key, benchmark in benchmarks.items()
        if key in submissions
    }


def calculate_score_weights(winner_percentage: float, scores: dict[Key, float]) -> dict[Key, float]:
    """
    Assumes that copies are removed from the scores
    """
    sorted_scores = sorted(scores.items(), key=itemgetter(1), reverse=True)

    return {
        hotkey: winner_percentage * ((1 - winner_percentage) ** index)
        for index, (hotkey, score) in enumerate(sorted_scores)
    }
