from operator import itemgetter

import numpy
from base.checkpoint import Key, Submissions, Benchmarks
from base.contest import Metrics
from base.inputs_api import get_inputs_state

DEVIATION_THRESHOLD_PERCENTILE = 90


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


def get_contestant_ranks(scores: dict[Key, float]) -> dict[Key, int]:
    if not scores:
        return {}

    rank = 0

    if len(scores) == 1:
        hotkey = next(iter(scores))

        return { hotkey: rank }

    i = 0

    scores = list(sorted(scores.items(), key=itemgetter(1), reverse=True))
    score_values = list(map(itemgetter(1), scores))

    deviations = numpy.array(list(
        score_values[i] - score_values[i + 1]
        for i in range(len(score_values) - 1)
        if score_values[i + 1] > 0
    ))

    if not len(deviations):
        return {}

    threshold = numpy.percentile(deviations, DEVIATION_THRESHOLD_PERCENTILE)

    scores = iter(scores)

    hotkey, last_score = next(scores)

    ranks = { hotkey: rank }

    for hotkey, score in scores:
        difference = last_score - score
        i += 1

        if difference > threshold:
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

    winner_percentage = get_inputs_state().winner_percentage
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
        if not hotkeys:
            continue
        incentive_pool = winner_percentage * ((1 - winner_percentage) ** index)
        score = incentive_pool / len(hotkeys)

        for hotkey in hotkeys:
            weights[hotkey] = score

    return weights
