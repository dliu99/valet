import numpy as np
import pytest

from valet.armd.fake import FakeArm
from valet.armd.guard import Guard, NotEngaged
from valet.armd.kinematics import MOTORS

ELBOW = MOTORS.index("elbow_flex")
LIFT = MOTORS.index("shoulder_lift")


def ramp(guard: Guard, joint: int, degrees: float, ticks: int = 30):
    start = guard.goal.copy()
    out = []
    for k in range(1, ticks + 1):
        g = start.copy()
        g[joint] += degrees * k / ticks
        out.append(g)
    return out


def test_hold_sets_goal_to_present_and_engages(guard, arm):
    arm.relax()
    guard.hold()
    assert arm.torque_enabled()
    assert np.allclose(guard.goal, arm.q())


def test_a_motion_needs_a_hold_first(guard):
    with pytest.raises(NotEngaged):
        guard.follow([np.zeros(6)])


class Sagging(FakeArm):
    def q(self):
        q = super().q()
        q[LIFT] -= 2.0
        return q


def test_holding_again_keeps_the_goal_so_sag_does_not_compound(kin, clock):
    arm = Sagging(kin, clock)
    guard = Guard(arm, clock)
    guard.hold()
    clock.sleep(0.5)
    settled = arm.q()
    guard.hold()
    clock.sleep(0.5)
    assert np.allclose(arm.q(), settled)


def test_relax_interrupts_a_motion(guard, arm):
    guard.hold()

    def path():
        for i, g in enumerate(ramp(guard, ELBOW, 20.0)):
            if i == 5:
                guard.relax("relaxed by a person")
            yield g

    out = guard.follow(path())
    assert out.stopped == "relaxed by a person"
    assert not arm.torque_enabled() and guard.goal is None
