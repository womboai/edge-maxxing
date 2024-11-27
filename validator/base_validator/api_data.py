from enum import Enum

from pydantic import BaseModel

from base.checkpoint import Key, Uid, Benchmarks
from base.contest import ContestId, RepositoryInfo


class ApiMetadata(BaseModel):
    version: str
    compatible_contests: set[ContestId]


class BenchmarkState(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    FINISHED = 2


class BenchmarkingResults(BaseModel):
    state: BenchmarkState
    benchmarks: Benchmarks
    invalid_submissions: set[Key]
    average_benchmark_time: float | None


class BenchmarkingStartRequest(BaseModel):
    contest_id: ContestId
    submissions: dict[Key, RepositoryInfo]


class BenchmarkingInitializeRequest(BaseModel):
    uid: Uid
    signature: str
    substrate_url: str
