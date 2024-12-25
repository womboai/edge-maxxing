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


def calculate_score_weights(_winner_percentage: float, scores: dict[Key, float]) -> dict[Key, float]:
    """
    Assumes that copies are removed from the scores
    """

    min_score = min(scores.values())

    return {
        hotkey: score + min_score
        for hotkey, score in scores
    }
