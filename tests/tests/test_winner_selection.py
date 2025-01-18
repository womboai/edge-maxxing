from unittest import TestCase

from base.contest import MetricType, Contest, Metrics, Benchmark

from weight_setting.winner_selection import calculate_score_weights

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
                size=1,
                vram_used=vram,
                watts_used=1,
                load_time=1,
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
            time=0.01,
            vram=10,
            ram=0.01,
        )

        bad_model = metrics(
            time=10,
            vram=10,
            ram=0.01,
        )

        self.assertEqual(score(baseline, 1.0), 0.0)
        self.assertGreater(score(perfect_model, 1.0), score(bad_model, 1.0))
        self.assertEqual(score(perfect_model, 0.0), -1.0)

        # Time is more important than VRAM in the weights set above, so slightly faster time is better than significantly better VRAM(which according to the weights, higher is better)
        self.assertGreater(score(metrics(time=3, vram=5, ram=5)), score(metrics(time=5, vram=9, ram=5)))
        self.assertGreater(score(metrics(time=3, vram=4, ram=5)), score(metrics(time=4, vram=5, ram=5)))

    def test_score_weights(self):
        scores = {
            "a": 0.212,
            "b": 0.21,
            "c": 0.19,
            "d": 0.15,
        }

        winner_percentage = 0.25
        scores = calculate_score_weights(winner_percentage, scores)

        self.assertEqual(scores["a"], winner_percentage)
        self.assertGreater(scores["a"], scores["b"])
        self.assertGreater(scores["b"], scores["c"])
        self.assertGreater(scores["c"], scores["d"])
