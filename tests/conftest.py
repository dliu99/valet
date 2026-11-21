from pathlib import Path

import pytest

from valet.armd.app import Armd, open_fake
from valet.armd.fake import FakeArm
from valet.armd.guard import Guard
from valet.armd.kinematics import Kinematics
from valet.armd.settings import Settings


class FastClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


@pytest.fixture
def clock() -> FastClock:
    return FastClock()


@pytest.fixture
def kin() -> Kinematics:
    return Kinematics()


@pytest.fixture
def arm(kin, clock) -> FakeArm:
    return FakeArm(kin, clock)


@pytest.fixture
def guard(arm, clock) -> Guard:
    return Guard(arm, clock)


@pytest.fixture
def make_app(tmp_path: Path, clock: FastClock):
    def make(**settings) -> Armd:
        return open_fake(Settings(**{"allow_motion": True, **settings}), clock=clock, poses_path=tmp_path / "poses.json")

    return make


@pytest.fixture
def app(make_app) -> Armd:
    a = make_app()
    a.hold()
    return a
