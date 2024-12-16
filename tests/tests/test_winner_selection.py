import json
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from unittest import TestCase

from base.checkpoint import Uid, Key
from base.contest import MetricType, Contest, Metrics, Benchmark
from weight_setting.winner_selection import get_contestant_ranks, calculate_rank_weights

TEST_DATA_DIRECTORY = Path(__file__).parent.parent / "test_data"

class WinnerSelectionTest(TestCase):
    def test_scoring(self):
        class MockContest:
            def get_metric_weights(self):
                return {
                    MetricType.GENERATION_TIME: 3,
                    MetricType.RAM_USED: 1,
                    MetricType.VRAM_USED: -1,
                }

            def calculate_score(self, baseline: Metrics, benchmark: Benchmark):
                return Contest.calculate_score(self, baseline, benchmark)  # type: ignore

        contest = MockContest()

        def metrics(time: float, vram: float, ram: float):
            return Metrics(
                generation_time=time,
                size=0,
                vram_used=vram,
                watts_used=0,
                load_time=0,
                ram_used=ram,
            )

        def score(metrics: Metrics, similarity: float = 1.0):
            return contest.calculate_score(baseline, Benchmark(metrics=metrics, average_similarity=similarity, min_similarity=similarity))

        baseline = metrics(
            time=5,
            vram=5,
            ram=5,
        )

        perfect_model = metrics(
            time=0,
            vram=float('inf'),  # or 2 * baseline
            ram=0,
        )

        bad_model = metrics(
            time=10,
            vram=10,
            ram=0,
        )

        self.assertEqual(score(baseline, 1.0), 0.0)
        self.assertEqual(score(perfect_model, 1.0), 1.0)
        self.assertEqual(score(perfect_model, 0.0), -1.0)
        self.assertEqual(score(bad_model, 1.0), -1.0)

        # Time is more important than VRAM in the weights set above, so slightly faster time is better than significantly better VRAM(which according to the weights, higher is better)
        self.assertGreater(score(metrics(time=3, vram=5, ram=5)), score(metrics(time=5, vram=9, ram=5)))
        self.assertGreater(score(metrics(time=3, vram=4, ram=5)), score(metrics(time=4, vram=5, ram=5)))

    def test_winner_selection(self):
        for path in TEST_DATA_DIRECTORY.glob("*.json"):
            winners: dict[Key, list[Uid]] = defaultdict(list)
            date: datetime = datetime.strptime(path.stem, "%Y-%m-%d")
            with path.open("r") as file:
                for uid, data in json.load(file).items():
                    scores = {key: info["score"] for key, info in data.items()}
                    submitted_blocks = {key: info["block"] for key, info in data.items()}

                    ranks = get_contestant_ranks(scores)
                    weights = calculate_rank_weights(submitted_blocks, ranks)

                    winner = max(
                        weights.items(),
                        key=itemgetter(1),
                        default=None
                    )

                    if not winner:
                        continue

                    winners[winner[0]].append(uid)

            msg = f"Multiple winners found on {date.strftime('%Y-%m-%d')}:\n" + "\n".join(
                f"{winner}: {', '.join(map(str, validator_uids))}"
                for winner, validator_uids in winners.items()
            )

            self.assertEqual(len(winners), 1, msg=msg)
