from datetime import datetime, timedelta
from threading import Event

from fiber.logging_utils import get_logger
from pydantic import BaseModel, ConfigDict

from base.checkpoint import Key, current_time, Submissions, Benchmarks
from base.contest import Metrics, BenchmarkState
from weight_setting.winner_selection import get_contestant_scores, get_contestant_ranks, calculate_rank_weights

logger = get_logger(__name__)


class ContestState(BaseModel):
    model_config = ConfigDict(extra="ignore")
    step: int
    benchmarks_version: int
    submissions: Submissions
    benchmarks: Benchmarks
    baseline: Metrics | None
    invalid_submissions: set[Key]
    last_benchmarks: Benchmarks
    average_benchmarking_time: float | None
    benchmarking_state: BenchmarkState
    contest_end: datetime

    def start_new_contest(self, benchmarks_version: int, submissions: Submissions):
        logger.info("Starting a new contest")
        self.benchmarks_version = benchmarks_version
        self.submissions = submissions

        if self.benchmarking_state == BenchmarkState.FINISHED:
            logger.info("Updating benchmarks for weight setting")
            self.last_benchmarks = self.benchmarks

        self.benchmarks.clear()
        self.invalid_submissions.clear()
        self.baseline = None
        self.average_benchmarking_time = None

        now = current_time()
        end_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now.hour >= 12:
            end_time += timedelta(days=1)
        self.contest_end = end_time
        self.benchmarking_state = BenchmarkState.NOT_STARTED

    def get_contest_start(self) -> datetime:
        return self.contest_end - timedelta(days=1)

    def is_ended(self) -> bool:
        now = current_time()
        return now >= self.contest_end or not self.submissions

    def get_untested_submissions(self) -> Submissions:
        return {
            key: submission for key, submission in self.submissions.items()
            if key not in self.benchmarks and key not in self.invalid_submissions
        }

    def sleep_to_next_contest(self, stop_flag: Event):
        now = current_time()
        next_contest_time = self.contest_end - now
        logger.info(f"Sleeping until next contest: {next_contest_time}")
        stop_flag.wait(next_contest_time.total_seconds())

    def get_scores(self, benchmarks: Benchmarks) -> dict[Key, float]:
        if not self.baseline:
            return {}

        return get_contestant_scores(
            submissions=self.submissions,
            benchmarks=benchmarks,
            baseline=self.baseline,
        )

    def get_ranks(self, scores: dict[Key, float]) -> dict[Key, int]:
        return get_contestant_ranks(scores=scores)

    def calculate_weights(self, ranks: dict[Key, int]):
        return calculate_rank_weights(
            submitted_blocks={
                key: submission.block
                for key, submission in self.submissions.items()
            },
            ranks=ranks,
        )

    @classmethod
    def create(cls, benchmarks_version: int):
        state = cls(
            step=0,
            benchmarks_version=benchmarks_version,
            submissions={},
            benchmarks={},
            baseline=None,
            invalid_submissions=set(),
            last_benchmarks={},
            average_benchmarking_time=None,
            benchmarking_state=BenchmarkState.NOT_STARTED,
            contest_end=current_time()
        )
        return state
