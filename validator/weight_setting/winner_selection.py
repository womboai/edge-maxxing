from operator import itemgetter
from statistics import median

from base.checkpoint import Key, Submissions, Benchmarks
from base.contest import Metrics

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

    i = 0
    rank = 0

    scores = list(sorted(scores.items(), key=itemgetter(1), reverse=True))
    score_values = list(map(itemgetter(1), scores))

    deviation = median(score_values[i] - score_values[i + 1] for i in range(len(score_values) - 1))

    scores = iter(scores)

    hotkey, last_score = next(scores)

    ranks = {hotkey: rank}

    for hotkey, score in scores:
        difference = last_score - score
        i += 1

        if difference > deviation:
            deviation = (deviation * i + difference) / i
            rank += 1

        ranks[hotkey] = rank
        last_score = score

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
