"""A learned skill: a LeRobot policy (ACT, SmolVLA, ...) run for a fixed number of steps.

Train one narrow policy per skill ("pick a slice from the table", "place what is held on
the stack") from teleoperated demos recorded with the SO-101 leader arm. The camera
names and resolution must match the training dataset's; check its info.json.

Untested against hardware: written from LeRobot's examples/tutorial/act/act_using_example.py.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .skills import SkillContext, SkillResult


@dataclass
class LeRobotPolicySkill:
    policy_path: str  # a local checkpoint directory or a Hub repo id
    dataset_repo_id: str  # the dataset it was trained on: features and normalization stats
    task: str = ""  # language instruction, for policies that take one
    max_steps: int = 300
    fps: float = 30.0
    device: str = "cpu"  # "cuda" or "mps" where available
    _loaded: dict[str, Any] = field(default_factory=dict, repr=False)

    def _load(self) -> dict[str, Any]:
        if self._loaded:
            return self._loaded
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets import LeRobotDatasetMetadata
        from lerobot.policies import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy = get_policy_class(cfg.type).from_pretrained(self.policy_path)
        policy.to(self.device).eval()
        meta = LeRobotDatasetMetadata(self.dataset_repo_id)
        pre, post = make_pre_post_processors(policy.config, dataset_stats=meta.stats)
        self._loaded = {"torch": torch, "policy": policy, "pre": pre, "post": post, "features": meta.features}
        return self._loaded

    def run(self, robot, ctx: SkillContext) -> SkillResult:
        lerobot_robot = getattr(robot, "lerobot", None)
        if lerobot_robot is None:
            return SkillResult(False, "a LeRobot policy needs the SO-101 robot, not this one")
        from lerobot.policies.utils import build_inference_frame, make_robot_action

        m = self._load()
        torch, policy = m["torch"], m["policy"]
        policy.reset()
        period = 1.0 / self.fps
        with torch.inference_mode():
            for _ in range(self.max_steps):
                start = time.perf_counter()
                obs = lerobot_robot.get_observation()
                frame = build_inference_frame(obs, torch.device(self.device), m["features"], task=self.task)
                out = policy.select_action(m["pre"](frame))
                action = out["action"] if isinstance(out, Mapping) else out
                robot.send_action(make_robot_action(m["post"](action), m["features"]))
                time.sleep(max(0.0, period - (time.perf_counter() - start)))
        return SkillResult(True, f"ran {self.max_steps} policy steps")
