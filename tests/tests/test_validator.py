from unittest.mock import Mock

import pytest
from base_validator.metrics import CheckpointBenchmark, MetricData
from neuron import MinerModelInfo, ModelRepositoryInfo
from weight_setting.validator import Validator, ContestState


@pytest.fixture
def mock_benchmarks() -> list[CheckpointBenchmark | None]:
    def create(baseline_generation_time, generation_time, similarity_score) -> CheckpointBenchmark:
        return CheckpointBenchmark(
            baseline=Mock(spec=MetricData, generation_time=baseline_generation_time),
            model=Mock(spec=MetricData, generation_time=generation_time),
            similarity_score=similarity_score
        )

    return [
        None,
        create(baseline_generation_time=2.5, generation_time=2.5, similarity_score=1.0),  # baseline submission
        create(baseline_generation_time=2.5, generation_time=2.5, similarity_score=0.0),  # no similarity
        create(baseline_generation_time=2.5, generation_time=10, similarity_score=0.5),  # bad submission
        create(baseline_generation_time=2.5, generation_time=0.9, similarity_score=0.85),  # reasonable submission
        create(baseline_generation_time=2.5, generation_time=0.5, similarity_score=0.75),  # top submission
        create(baseline_generation_time=2.5, generation_time=0.485, similarity_score=0.76),  # top submission with slightly higher score
        None,
    ]


@pytest.fixture
def mock_miner_info() -> list[MinerModelInfo | None]:
    def create(block) -> MinerModelInfo:
        return Mock(spec=MinerModelInfo, repository=Mock(spec=ModelRepositoryInfo), block=block)

    return [
        None,
        create(block=10),
        create(block=10),
        create(block=10),
        create(block=10),
        create(block=10),
        create(block=100),  # submitted at a later point
        None,
    ]


class TestValidator:
    @pytest.fixture(autouse=True)
    def init(self, mock_benchmarks, mock_miner_info):
        self.benchmarks = mock_benchmarks

        self.contest_state = Mock(spec=ContestState)
        self.contest_state.miner_info = mock_miner_info

    def test_score(self):
        scores = [benchmark.calculate_score() for benchmark in self.benchmarks if benchmark]
        assert scores == [0.0, 0.0, -3.75, 1.36, 1.5, 1.5314], f"Scores do not match: {scores}"

    def test_winner(self):
        winner = Validator.get_winner.__get__(self)()

        # 5 because 6 has a higher score but is below the improvement threshold and submitted later
        assert winner == 5, f"Winner does not match: {winner}"
