from datetime import datetime
from typing import TypeAlias
from zoneinfo import ZoneInfo

from base.contest import Submission, Benchmark

TIMEZONE = ZoneInfo("America/Los_Angeles")
SPEC_VERSION = 7

Uid: TypeAlias = int
Key: TypeAlias = str

Submissions: TypeAlias = dict[Key, Submission]
Benchmarks: TypeAlias = dict[Key, Benchmark]


def current_time() -> datetime:
    return datetime.now(tz=TIMEZONE)
