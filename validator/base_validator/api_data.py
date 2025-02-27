from pydantic import BaseModel

from base.checkpoint import Key, Uid, Benchmarks
from base.contest import ContestId, RepositoryInfo, BenchmarkState, Metrics


class ApiMetadata(BaseModel):
    version: str
    compatible_contests: set[ContestId]


class BenchmarkingResults(BaseModel):
    state: BenchmarkState
    benchmarks: Benchmarks
    invalid_submissions: set[Key]
    baseline: Metrics | None
    average_benchmarking_time: float | None


class BenchmarkingStartRequest(BaseModel):
    contest_id: ContestId
    submissions: dict[Key, RepositoryInfo]


class BenchmarkingInitializeRequest(BaseModel):
    uid: Uid
    signature: str
    netuid: Uid
    substrate_url: str
