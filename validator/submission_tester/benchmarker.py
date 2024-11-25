from concurrent.futures import CancelledError
from random import choice
from threading import Lock, Event, Thread
from time import perf_counter, sleep
from typing import cast

from fiber.logging_utils import get_logger
from opentelemetry import trace

from neuron import (
    Key,
    ModelRepositoryInfo,
    random_inputs,
    Contest,
)
from neuron.submission_tester import (
    CheckpointBenchmark,
    BaselineBenchmark,
    MetricData,
    compare_checkpoints,
    generate_baseline,
    InvalidSubmissionError,
)
from pipelines import TextToImageRequest

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class Benchmarker:
    contest: Contest | None
    submissions: dict[Key, ModelRepositoryInfo]
    benchmarks: dict[Key, CheckpointBenchmark | None]
    invalid: dict[Key, str]
    baseline: BaselineBenchmark | None
    inputs: list[TextToImageRequest]
    started: bool
    done: bool
    start_timestamp: int
    submission_times: list[float]
    thread: Thread | None
    lock: Lock
    cancelled_event: Event

    def __init__(self):
        self.contest = None
        self.submissions = {}
        self.benchmarks = {}
        self.invalid = {}
        self.baseline = None
        self.inputs = []
        self.done = False
        self.start_timestamp = 0
        self.thread = None
        self.lock = Lock()
        self.submission_times = []
        self.cancelled_event = Event()

    def _benchmark_submission(self, hotkey: Key):
        with tracer.start_as_current_span("benchmark_submission") as span:
            submission = self.submissions[hotkey]
            span.set_attributes({
                "submission.hotkey": str(hotkey),
                "submission.url": submission.url,
                "submission.revision": submission.revision
            })

            start_time = perf_counter()

            try:
                self.benchmarks[hotkey] = compare_checkpoints(
                    contest=self.contest,
                    submission=submission,
                    inputs=self.inputs,
                    baseline=self.baseline,
                    load_timeout=int(cast(MetricData, self.get_baseline_metrics()).load_time * 2),
                    cancelled_event=self.cancelled_event,
                )
            except InvalidSubmissionError as e:
                logger.error(f"Skipping invalid submission '{submission}': '{e}'")
                self.benchmarks[hotkey] = None
                self.invalid[hotkey] = str(e)
            except CancelledError:
                logger.warning(f"Benchmarking was canceled while testing '{submission}'")
                raise
            except Exception as e:
                span.record_exception(e)
            finally:
                duration = perf_counter() - start_time
                self.submission_times.append(duration)
                span.set_attribute("benchmark.duration_seconds", duration)

    def _start_benchmarking(self, contest: Contest, submissions: dict[Key, ModelRepositoryInfo]):
        with tracer.start_as_current_span("start_benchmarking") as span:
            span.set_attribute("submissions.total", len(submissions))

            self.contest = contest
            self.submissions = submissions
            self.benchmarks.clear()
            self.invalid.clear()
            self.submission_times.clear()
            self.inputs = random_inputs()
            self.done = False
            self.baseline = None

            with tracer.start_span("generate_baseline"):
                while not self.baseline and not self.cancelled_event.is_set():
                    try:
                        logger.info("Generating baseline samples to compare")
                        self.baseline = generate_baseline(
                            contest=contest,
                            inputs=self.inputs,
                            cancelled_event=self.cancelled_event
                        )
                    except CancelledError:
                        logger.warning("Benchmarking was canceled while testing the baseline")
                        return
                    except Exception as e:
                        logger.error("Failed to generate baseline samples, retrying in 10 minutes", exc_info=e)
                        sleep(600)

            with tracer.start_span("benchmark_submissions") as bench_span:
                while len(self.benchmarks) != len(self.submissions) and self.baseline and not self.cancelled_event.is_set():
                    hotkey = choice(list(self.submissions.keys() - self.benchmarks.keys()))
                    self._benchmark_submission(hotkey)

                    valid_submissions = len([benchmark for benchmark in self.benchmarks.values() if benchmark])
                    bench_span.set_attributes({
                        "progress.total": len(self.benchmarks),
                        "progress.valid": valid_submissions,
                    })
                    logger.info(f"{len(self.benchmarks)}/{len(self.submissions)} submissions benchmarked. {valid_submissions} valid.")

            logger.info("Benchmarking complete")
            self.done = True

    @tracer.start_as_current_span("start_benchmarking")
    def start_benchmarking(self, contest: Contest, submissions: dict[Key, ModelRepositoryInfo]):
        if not submissions:
            logger.warning("No submissions to benchmark")
            return

        logger.info(f"Starting benchmarking for {len(submissions)} submissions")

        if self.thread and self.thread.is_alive():
            logger.info("Attempting to cancel previous benchmarking")
            self.cancelled_event.set()
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                logger.warning("Benchmarking was not stopped gracefully.")
            else:
                logger.info("Benchmarking was stopped gracefully.")

        self.cancelled_event.clear()
        self.thread = Thread(
            target=self._start_benchmarking,
            args=(contest, submissions,),
            daemon=True,
        )
        self.thread.start()

    def get_baseline_metrics(self) -> MetricData | None:
        return self.baseline.metric_data if self.baseline else None
