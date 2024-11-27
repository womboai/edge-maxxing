from operator import itemgetter

from base.checkpoint import Key, Submissions, Benchmarks
from base.contest import Metrics

RANK_SCORE_IMPROVEMENT_THRESHOLD = 1.05
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


def get_contestant_ranks(scores: dict[Key, float]) -> dict[Key, int]:
    if not scores:
        return {}

    scores = iter(sorted(scores.items(), key=itemgetter(1), reverse=True))

    hotkey, last_score = next(scores)

    rank = 0
    ranks = {hotkey: rank}

    for hotkey, score in scores:
        if last_score > score * RANK_SCORE_IMPROVEMENT_THRESHOLD:
            last_score = score
            rank += 1

        ranks[hotkey] = rank

    return ranks


def calculate_rank_weights(
    submitted_blocks: dict[Key, int],
    ranks: dict[Key, int],
) -> dict[Key, float]:
    if not ranks:
        return {}

    ranks = iter(sorted(ranks.items(), key=lambda rank: (rank[1], submitted_blocks[rank[0]])))

    last_rank = None

    rank_hotkeys = [[]]

    for hotkey, rank in ranks:
        rank_hotkeys[-1].append(hotkey)

        if rank != last_rank:
            rank_hotkeys.append([])

            last_rank = rank

    weights = {}

    for index, hotkeys in enumerate(rank_hotkeys):
        incentive_pool = WINNER_PERCENTAGE * ((1 - WINNER_PERCENTAGE) ** index)
        score = incentive_pool / len(hotkeys)

        for hotkey in hotkeys:
            weights[hotkey] = score

    return weights
