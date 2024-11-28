from concurrent.futures import CancelledError
from datetime import timedelta
from pathlib import Path
from random import choice
from statistics import mean
from threading import Event, Thread
from time import perf_counter

from fiber.logging_utils import get_logger
from opentelemetry import trace

from base.checkpoint import Key, Benchmarks
from base.contest import Contest, RepositoryInfo, Benchmark, BenchmarkState
from base.inputs_api import random_inputs
from pipelines import TextToImageRequest
from .inference_sandbox import BenchmarkOutput, InferenceSandbox

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class Benchmarker:
    _thread: Thread | None = None
    _stop_flag: Event = Event()

    _sandbox_directory: Path
    _sandbox_args: list[str]

    start_timestamp: float = 0
    state: BenchmarkState = BenchmarkState.NOT_STARTED
    benchmarks: Benchmarks = {}
    invalid_submissions: set[Key] = set()
    baseline: BenchmarkOutput | None = None
    submission_times: list[float] = []

    def __init__(self, sandbox_directory: Path, sandbox_args: list[str]):
        self._sandbox_directory = sandbox_directory
        self._sandbox_args = sandbox_args

    @tracer.start_as_current_span("compare")
    def compare(
        self,
        contest: Contest,
        benchmark_output: BenchmarkOutput,
        stop_flag: Event,
    ) -> Benchmark:
        similarities: list[float] = []
        with contest.output_comparator() as comparator:
            for baseline_output, optimized_output in zip(self.baseline.outputs, benchmark_output.outputs):
                if stop_flag.is_set():
                    raise CancelledError()

                similarities.append(comparator(baseline_output, optimized_output))

        average_similarity = mean(similarities)
        min_similarity = min(similarities)

        return Benchmark(
            metrics=benchmark_output.metrics,
            average_similarity=average_similarity,
            min_similarity=min_similarity,
        )

    def _is_done(self, submissions: dict[Key, RepositoryInfo]) -> bool:
        return len(self.benchmarks) + len(self.invalid_submissions) == len(submissions)

    @tracer.start_as_current_span("benchmark")
    def _benchmark_submission(
        self,
        contest: Contest,
        inputs: list[TextToImageRequest],
        repository_info: RepositoryInfo
    ) -> BenchmarkOutput:
        return InferenceSandbox(
            sandbox_args=self._sandbox_args,
            sandbox_directory=self._sandbox_directory,
            contest=contest,
            repository_info=repository_info,
            inputs=inputs,
            stop_flag=self._stop_flag,
        ).benchmark()

    @tracer.start_as_current_span("benchmark_baseline")
    def _benchmark_baseline(self, contest: Contest, inputs: list[TextToImageRequest]):
        logger.info("Benchmarking baseline")
        while not self.baseline and not self._stop_flag.is_set():
            try:
                self._benchmark_submission(contest, inputs, contest.baseline_repository)
            except CancelledError:
                logger.warning("Benchmarking was canceled while testing the baseline")
                return
            except Exception as e:
                logger.error("Failed to generate baseline samples, retrying in 10 minutes", exc_info=e)
                self._stop_flag.wait(600)

    def benchmark_submissions(self, contest: Contest, submissions: dict[Key, RepositoryInfo]):
        self.state = BenchmarkState.IN_PROGRESS
        self.benchmarks.clear()
        self.invalid_submissions.clear()
        self.baseline = None
        self.submission_times.clear()

        inputs = random_inputs()
        self._benchmark_baseline(contest, inputs)

        logger.info(f"Benchmarking {len(submissions)} submissions")
        while not self._is_done(submissions) and self.baseline and not self._stop_flag.is_set():
            start_time = perf_counter()
            key = choice(list(submissions.keys() - self.benchmarks.keys()))
            submission = submissions[key]
            logger.info(f"Benchmarking submission '{submission.url}' with revision '{submission.revision}'")
            try:
                benchmark_output = self._benchmark_submission(contest, inputs, submission)
                benchmark = self.compare(contest, benchmark_output, self._stop_flag)
                self.benchmarks[key] = benchmark
            except CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to benchmark submission '{submission}': '{e}'", exc_info=e)
                self.invalid_submissions.add(key)
            finally:
                self.submission_times.append(perf_counter() - start_time)

            average_benchmark_time = self.get_average_benchmark_time()
            if average_benchmark_time:
                eta = (len(submissions) - len(self.benchmarks)) * average_benchmark_time
                logger.info(f"Average benchmark time: {average_benchmark_time}, ETA: {timedelta(seconds=eta)}")

        if self._is_done(submissions):
            logger.info("Benchmarking complete")
            self.state = BenchmarkState.FINISHED
        else:
            logger.warning("Benchmarking canceled")
            self.state = BenchmarkState.NOT_STARTED

    def get_average_benchmark_time(self) -> float | None:
        return (
            sum(self.submission_times) / len(self.submission_times)
            if self.submission_times
            else None
        )

    def shutdown(self):
        if self._thread and self._thread.is_alive():
            logger.info("Attempting to cancel previous benchmarking")
            self._stop_flag.set()
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                logger.warning("Benchmarking was not stopped gracefully.")
            else:
                logger.info("Benchmarking was stopped gracefully.")

    def start_benchmarking(self, contest: Contest, submissions: dict[Key, RepositoryInfo]):
        if not submissions:
            logger.warning("No submissions to benchmark")
            return

        logger.info(f"Started benchmarking for {len(submissions)} submissions")

        self.shutdown()

        self._stop_flag.clear()
        self._thread = Thread(
            target=self.benchmark_submissions,
            args=(contest, submissions,),
            daemon=True,
        )
        self._thread.start()
