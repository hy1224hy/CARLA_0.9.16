"""Versioned configuration; all relative artifact paths resolve from this project."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
GROUPS = {"G0": ("off", False), "G1": ("plain", False),
          "G2": ("selective", False), "G3": ("off", True),
          "G4": ("selective", True), "G5": ("plain", True)}
GROUPS.update({"G6_random": ("random", False), "G7_strength": ("strength", False)})


@dataclass
class Task:
    duration: float = 20.0
    policy_frequency: int = 5
    simulation_frequency: int = 15
    goal_distance: float = 180.0
    sensing_range: float = 25.0
    communication_range: float = 80.0
    attack_weight: float = 0.0
    collision_penalty: float = 10.0
    offroad_penalty: float = 5.0
    ttc_threshold: float = 2.0
    research_protocol: str = "legacy_v2"
    red_lane_change: bool = False
    red_target_speed: float = 18.0
    normal_blue_lane_change: bool = True
    normal_blue_target_speed: float = 18.0
    blue_speed_limit: float = 22.0
    overspeed_grace_s: float = 1.0
    selection_min_qualified: float = 0.75
    bottleneck: dict | None = None


@dataclass
class Config:
    schema: str = "highway_mut_mappo_v2"
    group: str = "G0"
    seed: int = 42
    total_train_steps: int = 100000
    rollout_steps: int = 512
    epochs: int = 4
    minibatch: int = 128
    hidden: int = 128
    message_dim: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    mutual_beta: float = 0.02
    warmup_rounds: int = 2
    refresh_rounds: int = 5
    gate_cases_per_family: int = 8
    gate_margin: float = 0.02
    gate_max_bad_rate: float = 0.125
    threads: int = 1
    backend: str = "highway"
    backend_contract: dict = field(default_factory=dict)
    exploration: dict | None = None
    curriculum: dict | None = None
    task: Task = field(default_factory=Task)

    @property
    def method(self):
        return GROUPS[self.group][0]

    @property
    def communication(self):
        return GROUPS[self.group][1]

    def validate(self):
        if self.exploration is not None:
            e = self.exploration
            if set(e) != {'initial', 'minimum', 'maximum'} or any(
                    len(e[k]) != 2 or any(not isinstance(v, (int,float)) or not math.isfinite(v) for v in e[k]) for k in e):
                raise ValueError('exploration requires two finite values per action dimension')
            if any(not 0 < lo <= start <= hi for lo,start,hi in zip(e['minimum'],e['initial'],e['maximum'])):
                raise ValueError('exploration needs 0 < minimum <= initial <= maximum')
        if self.curriculum is not None:
            from .training.curriculum import validate_schedule
            validate_schedule(self.curriculum)
            if self.task.research_protocol == 'legacy_v2' or self.task.attack_weight != self.curriculum['weights'][-1]:
                raise ValueError('curriculum requires v3 and nominal final attack weight')
        if self.schema != "highway_mut_mappo_v2" or self.group not in GROUPS:
            raise ValueError("unsupported schema/group")
        if self.backend not in ("highway", "carla"):
            raise ValueError("unsupported simulator backend")
        if self.backend == "carla":
            if self.task.research_protocol == "legacy_v2":
                raise ValueError("portable CARLA adapter requires the v3 contract")
            for key in ("scene_manifest_sha256", "red_controller_sha256", "normal_controller_sha256"):
                value = self.backend_contract.get(key, "")
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("CARLA requires explicit frozen backend contract hash: " + key)
        for name in ("total_train_steps", "rollout_steps", "epochs", "minibatch", "hidden",
                     "message_dim", "refresh_rounds", "gate_cases_per_family", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.warmup_rounds) is not int or self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be a nonnegative integer")
        for name in ("policy_frequency", "simulation_frequency"):
            if type(getattr(self.task, name)) is not int or getattr(self.task, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.task.duration * self.task.policy_frequency < 1 or self.task.ttc_threshold <= 0:
            raise ValueError("episode needs at least one decision and positive TTC threshold")
        if self.gate_cases_per_family > 100000:
            raise ValueError("case count would overlap reserved seed namespaces")
        if self.total_train_steps % 2:
            raise ValueError("A/B require an even total train-step budget")
        if min(self.task.duration, self.task.goal_distance, self.task.sensing_range,
               self.task.communication_range, self.task.policy_frequency) <= 0:
            raise ValueError("task durations/ranges/frequencies must be positive")
        if self.task.simulation_frequency % self.task.policy_frequency:
            raise ValueError("simulation frequency must be a multiple of policy frequency")
        for value in self.to_dict().values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite configuration")
        if any(not math.isfinite(v) for v in asdict(self.task).values() if isinstance(v, (int, float))):
            raise ValueError("nonfinite task configuration")
        if self.task.research_protocol not in ("legacy_v2", "route_relative_paired_attack_v3", "bottleneck_paired_attack_v4"):
            raise ValueError("unknown research protocol")
        if self.task.research_protocol == 'bottleneck_paired_attack_v4':
            from .bottleneck.contract import validate_geometry
            validate_geometry(self.task)
        elif self.task.bottleneck is not None:
            raise ValueError('bottleneck geometry requires the v4 protocol')
        if type(self.task.red_lane_change) is not bool or min(self.task.red_target_speed, self.task.blue_speed_limit) <= 0 or self.task.overspeed_grace_s < 0:
            raise ValueError("invalid traffic/safety contract")
        if type(self.task.normal_blue_lane_change) is not bool or self.task.normal_blue_target_speed <= 0:
            raise ValueError("invalid frozen normal-blue controller")
        if not 0 <= self.task.selection_min_qualified <= 1:
            raise ValueError("qualified selection threshold must be in [0,1]")
        if self.task.research_protocol != "legacy_v2":
            n = round(self.task.duration * self.task.policy_frequency)
            if abs(n - self.task.duration * self.task.policy_frequency) > 1e-6:
                raise ValueError("v3 fixed horizon must contain an integer number of policy decisions")
            if min(self.task.collision_penalty, self.task.offroad_penalty) * self.gamma ** (n - 1) <= 0.1:
                raise ValueError("discounted failure cost must exceed the total positive speed bonus")
        if min(self.lr, self.clip) <= 0 or min(self.entropy_coef, self.mutual_beta,
                self.warmup_rounds, self.gate_margin, self.task.attack_weight,
                self.task.collision_penalty, self.task.offroad_penalty) < 0:
            raise ValueError("invalid learning/reward magnitude")
        if not 0 <= self.gate_max_bad_rate <= 1:
            raise ValueError("gate bad-rate threshold must be in [0,1]")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid GAE discount")
        return self

    def to_dict(self):
        result = asdict(self)
        if result['task']['bottleneck'] is None:
            result['task'].pop('bottleneck')
        for name in ('exploration', 'curriculum'):
            if result[name] is None:
                result.pop(name)
        if self.backend == "highway" and not self.backend_contract:
            result.pop("backend")
            result.pop("backend_contract")
        if self.task.research_protocol == "legacy_v2":
            for name in ("research_protocol", "red_lane_change", "red_target_speed", "normal_blue_lane_change", "normal_blue_target_speed", "blue_speed_limit", "overspeed_grace_s", "selection_min_qualified"):
                result["task"].pop(name)
        return result

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load(path=None):
    if path is None:
        return Config().validate()
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    data["task"] = Task(**data.get("task", {}))
    return Config(**data).validate()
