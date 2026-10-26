"""Thin four-car, two-blue-team learner over the existing MAPPO PPO path.

No CARLA calls are made here. A/B own their critic, optimizers, ValueNorm and
random streams; the frozen red policy never enters this module's optimizers.
Research thresholds remain configurable candidate values until pilot freeze.
"""

from __future__ import annotations

import copy
import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from models.actor_critic import Actor, AgentConditionedCritic
from models.mut_mappo_critics import LocalObservationCritic
from training.action_protocol import ACTION_PROTOCOL_VERSION
from training.checkpoint_protocol import capture_rng_state, restore_rng_state
from training.mutual_distillation import (
    build_distillation_gates, student_teacher_kl, validate_distillation_options,
)
from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer, build_active_step_cache, prepare_mappo_update_batch,
    run_mappo_ppo_update,
)
from training.value_normalization import ValueNormalizer


LEARNER_PROTOCOL = "mut_mappo_four_car_20_2_v1"
BLUE_IDS = (2, 3)
MODEL_SPEC = {"obs_dim": 20, "action_dim": 2, "global_state_dim": 80,
              "critic_id_dim": 4, "blue_physical_ids": [2, 3]}


def actor_states_hash(states: Mapping) -> str:
    """Stable tensor-content hash, independent of torch archive timestamps."""
    digest = hashlib.sha256()
    for aid in sorted(states, key=int):
        digest.update(str(aid).encode("ascii"))
        for name, tensor in sorted(states[aid].items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _single_role_buffer(buffer: RolloutBuffer, aid: int) -> RolloutBuffer:
    """Preserve a role's episode slicing and bootstrap evidence unchanged."""
    result = RolloutBuffer([aid])
    result.data[aid] = list(buffer.data[aid])
    result.episode_boundaries[aid] = list(buffer.episode_boundaries[aid])
    result.episode_bootstrap_states[aid] = list(buffer.episode_bootstrap_states[aid])
    result.scenario_names = list(buffer.scenario_names)
    return result


def _stats(stats: Mapping) -> tuple[int, float, float]:
    n, q, b = (stats[key] for key in ("n_all", "q_count", "b_count"))
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (n, q, b)):
        raise ValueError("gate statistics must be integer episode counts")
    if n <= 0 or not 0 <= q <= n or not 0 <= b <= n:
        raise ValueError("gate counts must use all valid episodes as the denominator")
    if q > n - b:
        raise ValueError("qualified response episodes cannot also be unsafe episodes")
    return int(n), q / n, b / n


@dataclass
class GateState:
    """Versioned, per-family paired gate evaluation; no test-set access."""
    config: dict = field(default_factory=dict)
    version: int = 0
    refresh_round: int = 0
    teacher_hashes: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    gates: dict = field(default_factory=lambda: {"A": {}, "B": {}})

    def compute(self, statistics, teacher_hashes, *, version, completed_round):
        """Runner adapter for episode-count metric names; never change N."""
        adapted = {}
        for team, families in statistics.items():
            adapted[team] = {}
            for family, record in families.items():
                converted = copy.deepcopy(record)
                for target, source in (("n_all", "n"), ("q_count", "eligible_response_count"),
                                       ("b_count", "violation_count")):
                    if target not in converted:
                        converted[target] = converted[source]
                converted.setdefault("teacher_hash", teacher_hashes[team])
                converted.setdefault("version", version)
                adapted[team][family] = converted
        return self.refresh(teacher_hashes, adapted, version=version, completed_round=completed_round)

    def refresh(self, teacher_hashes: dict, statistics: dict, *, version: int,
                completed_round: int) -> dict:
        if set(teacher_hashes) != {"A", "B"} or set(statistics) != {"A", "B"}:
            raise ValueError("gate refresh requires both frozen A/B teachers")
        if version <= self.version or completed_round < self.refresh_round:
            raise ValueError("gate refresh version/round must advance monotonically")
        if set(statistics["A"]) != set(statistics["B"]):
            raise ValueError("A/B gate families must match")
        minimum = int(self.config.get("min_samples", self.config.get("gate_min_cases", 8)))
        b_max = float(self.config.get("teacher_b_max", self.config.get("gate_max_bad_rate", 1 / 8)))
        q_margin = float(self.config.get("q_margin", self.config.get("gate_min_q_gap", 2 / 8)))
        if minimum < 1 or not 0 <= b_max <= 1 or not 0 <= q_margin <= 1:
            raise ValueError("invalid candidate gate thresholds")
        gates = {"A": {}, "B": {}}
        for family in statistics["A"]:
            for student, teacher in (("A", "B"), ("B", "A")):
                own, peer = statistics[student][family], statistics[teacher][family]
                ns, qs, bs = _stats(own)
                nt, qt, bt = _stats(peer)
                if ns != nt:
                    raise ValueError("gate A/B must have the same valid-case denominator")
                if own.get("case_keys") != peer.get("case_keys"):
                    raise ValueError("gate A/B paired case keys differ")
                for name, record in ((student, own), (teacher, peer)):
                    if record.get("teacher_hash") != teacher_hashes[name] or record.get("version") != version:
                        raise ValueError("gate statistics mismatch frozen teacher identity/version")
                gates[student][family] = int(
                    ns >= minimum and nt >= minimum and bt <= b_max
                    and bt <= bs and qt - qs >= q_margin
                )
        self.version, self.refresh_round = int(version), int(completed_round)
        self.teacher_hashes = copy.deepcopy(teacher_hashes)
        self.statistics, self.gates = copy.deepcopy(statistics), gates
        return copy.deepcopy(gates)

    def gates_for(self, student: str, peer_hash: str) -> dict:
        peer = "B" if student == "A" else "A" if student == "B" else None
        if peer is None or self.teacher_hashes.get(peer) != peer_hash:
            raise ValueError("gate does not describe this frozen peer")
        return copy.deepcopy(self.gates[student])

    def state_dict(self) -> dict:
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state: Mapping) -> None:
        if set(state) != set(vars(self)) or state["config"] != self.config:
            raise ValueError("gate schema/config differs on resume")
        replacement = GateState(copy.deepcopy(self.config))
        if state["version"]:
            replacement.refresh(state["teacher_hashes"], state["statistics"],
                                version=state["version"], completed_round=state["refresh_round"])
            if replacement.gates != state["gates"]:
                raise ValueError("stored gates differ from stored validation evidence")
        self.__dict__.update(copy.deepcopy(vars(replacement)))


class TeamLearner:
    """A single blue team; public IDs are physical IDs 2/3 throughout."""
    def __init__(self, name: str, seed: int, initial_actors: Mapping,
                 config: Mapping, device="cpu"):
        if name not in {"A", "B"}:
            raise ValueError("team name must be A or B")
        self.name, self.seed = name, int(seed)
        self.config, self.device = copy.deepcopy(dict(config)), torch.device(device)
        self.agent_ids = list(BLUE_IDS)
        self.algorithm = self.config.get("algorithm", "mappo")
        if self.algorithm not in {"mappo", "ippo"}:
            raise ValueError("algorithm must be mappo/ippo")
        self.method = self.config.get("method", "off")
        if self.method not in {"off", "plain", "selective"}:
            raise ValueError("method must be off/plain/selective")
        if self.algorithm == "ippo" and self.method != "off":
            raise ValueError("the independent IPPO baseline requires method=off")
        if not isinstance(self.config.get("distillation", {}), Mapping):
            raise ValueError("distillation configuration must be a mapping")
        self.distillation = validate_distillation_options(self.config.get("distillation", {}))
        if self.method != "selective" and self.distillation["gate_variant"] != "evidence":
            raise ValueError("random/strength gate controls require method=selective")
        self.beta = float(self.config.get("beta", 0.01))
        self.anchor_beta = float(self.config.get("anchor_beta", 0.0))
        if min(self.beta, self.anchor_beta) < 0 or not np.isfinite([self.beta, self.anchor_beta]).all():
            raise ValueError("auxiliary coefficients must be finite and nonnegative")
        source = {int(k): v for k, v in initial_actors.items()}
        if set(source) == {0, 1}:
            source = {2: source[0], 3: source[1]}
        if set(source) != set(BLUE_IDS):
            raise ValueError("initial actors must contain source roles 0/1 or physical IDs 2/3")
        hidden = int(self.config.get("hidden", self.config.get("hidden_dim", source[2]["net.0.weight"].shape[0])))
        self.model_spec = dict(MODEL_SPEC, hidden=hidden,
                               init_log_std=float(self.config.get("init_log_std", -0.5)),
                               min_log_std=float(self.config.get("min_log_std", -1.0)),
                               max_log_std=self.config.get("max_log_std"),
                               algorithm=self.algorithm,
                               critic_input_scope=("own_physical_slot_local_observation_only"
                                   if self.algorithm == "ippo" else "all_physical_observation_slots_plus_id"),
                               value_normalizer_scope=("independent_per_blue_role"
                                   if self.algorithm == "ippo" else "pooled_active_blue_role_returns"))
        global_rng = capture_rng_state()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.actors = {aid: self._new_actor() for aid in BLUE_IDS}
            for aid in BLUE_IDS:
                self.actors[aid].load_state_dict(source[aid], strict=True)
            # Deliberately never import the old two-car critic or ValueNorm.
            if self.algorithm == "ippo":
                self.critics = {aid: LocalObservationCritic(80, 4, hidden,
                    agent_ids=(aid,)).to(self.device) for aid in BLUE_IDS}
                self.value_normalizers = {aid: ValueNormalizer(device=self.device) for aid in BLUE_IDS}
                self.critic = self.value_normalizer = None
            else:
                self.critic = AgentConditionedCritic(80, 4, hidden).to(self.device)
                self.value_normalizer = ValueNormalizer(device=self.device)
                self.critics = {"shared": self.critic}
                self.value_normalizers = {"shared": self.value_normalizer}
            self._rng_state = capture_rng_state()
        finally:
            restore_rng_state(global_rng)
        self.actor_optims = {aid: torch.optim.Adam(actor.parameters(), lr=float(self.config.get("lr_actor", 3e-4)))
                             for aid, actor in self.actors.items()}
        self.critic_optims = {key: torch.optim.Adam(critic.parameters(),
            lr=float(self.config.get("lr_critic", 3e-4))) for key, critic in self.critics.items()}
        self.critic_optim = self.critic_optims.get("shared")
        self.actor_optimizers, self.critic_optimizer = self.actor_optims, self.critic_optim
        self.scenario_rng = random.Random(self.seed + 104729)
        self.anchor = self.frozen_actors()
        self.peer = None
        self.peer_snapshot = None
        self.peer_gates = {}
        self.update_count = 0
        self.cost_state = {"lambda": float(self.config.get("cost_lambda", 1.0)), "updates": 0}
        self.schedule_state = {"update": 0, "actor_lr": float(self.config.get("lr_actor", 3e-4)),
                               "critic_lr": float(self.config.get("lr_critic", 3e-4)),
                               "entropy": float(self.config.get("ent_coef", 0.03))}
        self._in_rng_context = False

    def _new_actor(self) -> Actor:
        spec = self.model_spec
        return Actor(20, 2, spec["hidden"], spec["init_log_std"],
                     spec["min_log_std"], spec["max_log_std"]).to(self.device)

    @contextmanager
    def rng_context(self):
        if self._in_rng_context:
            # Sampling helper may be called inside an outer team collection.
            yield
            return
        before = capture_rng_state()
        self._in_rng_context = True
        try:
            restore_rng_state(self._rng_state)
            yield
        finally:
            self._rng_state = capture_rng_state()
            restore_rng_state(before)
            self._in_rng_context = False

    def next_episode_seed(self) -> int:
        return self.scenario_rng.randrange(0, 2**31 - 1)

    def sample(self, obs_dict, global_state, active_mask):
        if set(active_mask) not in ({0, 1, 2, 3}, {2, 3}):
            raise ValueError("sampling mask must use all physical IDs or only blue IDs")
        state = np.asarray(global_state, dtype=np.float32)
        if state.shape != (80,) or not np.isfinite(state).all():
            raise ValueError("four-car global state must be finite shape (80,)")
        with self.rng_context():
            if self.algorithm == "ippo":
                actions, cache = {}, {}
                for aid in BLUE_IDS:
                    role_actions, role_cache = build_active_step_cache(
                        obs_dict, {aid: active_mask[aid]}, state, {aid: self.actors[aid]},
                        self.critics[aid], [aid], self.device,
                        context=f"Mut IPPO {self.name} role={aid} sample",
                        value_normalizer=self.value_normalizers[aid])
                    actions.update(role_actions)
                    cache.update(role_cache)
                return actions, cache
            return build_active_step_cache(obs_dict, {aid: active_mask[aid] for aid in BLUE_IDS},
                state, self.actors, self.critic, self.agent_ids, self.device,
                context=f"MutMAPPO {self.name} sample", value_normalizer=self.value_normalizer)

    def export_actors(self) -> dict:
        return {aid: {key: tensor.detach().cpu().clone() for key, tensor in actor.state_dict().items()}
                for aid, actor in self.actors.items()}

    def frozen_actors(self) -> dict:
        frozen = copy.deepcopy(self.actors)
        for actor in frozen.values():
            actor.eval().requires_grad_(False)
        return frozen

    def snapshot(self, version=0) -> dict:
        states = self.export_actors()
        return {"actors": states, "hash": actor_states_hash(states), "version": int(version),
                "model_spec": copy.deepcopy(self.model_spec), "team": self.name}

    def _compatible_model_spec(self, spec) -> bool:
        if spec == self.model_spec:
            return True
        # Legacy *default* MAPPO specs omitted these declarations. This is
        # learner decoding only; run config/source-identity checks still apply.
        if self.algorithm != "mappo" or "algorithm" in self.config:
            return False
        declarations = {"algorithm", "critic_input_scope", "value_normalizer_scope"}
        return spec == {key: value for key, value in self.model_spec.items() if key not in declarations}

    def set_peer(self, snapshot: Mapping | None, gates=None) -> None:
        if snapshot is None:
            self.peer = self.peer_snapshot = None
            self.peer_gates = {}
            return
        if not self._compatible_model_spec(snapshot.get("model_spec")):
            raise ValueError("peer input/action/distribution model specification differs")
        if snapshot.get("hash") != actor_states_hash(snapshot["actors"]):
            raise ValueError("frozen peer actor hash mismatch")
        before = capture_rng_state()
        try:
            peer = {aid: self._new_actor() for aid in BLUE_IDS}
            for aid in BLUE_IDS:
                peer[aid].load_state_dict(snapshot["actors"][aid], strict=True)
                peer[aid].eval().requires_grad_(False)
        finally:
            restore_rng_state(before)
        checked_gates = dict(gates or {})
        if any(g not in (0, 1) for g in checked_gates.values()):
            raise ValueError("peer family gates must be binary")
        self.peer, self.peer_snapshot = peer, copy.deepcopy(dict(snapshot))
        self.peer_gates = checked_gates

    def update(self, buffer: RolloutBuffer, aux_context=None) -> dict:
        if set(buffer.data) != set(BLUE_IDS):
            raise ValueError("team PPO buffer must contain only blue IDs 2/3")
        cfg = self.config
        aux_context = aux_context or {}
        family_map = aux_context.get("family_by_scenario", {})
        mutual_enabled = self.method != "off" and self.beta > 0 and self.peer is not None
        direction = self.distillation["direction"]
        reduction = self.distillation["reduction"]
        sample_gates = {}
        evidence_gates = {}
        aux_totals = {aid: {"n": 0, "raw": 0.0, "gated": 0.0, "anchor": 0.0} for aid in BLUE_IDS}
        for aid in BLUE_IDS:
            rows = buffer.get_agent_data(aid)
            evidence_gates[aid] = np.asarray([
                1 if self.method == "plain" else self.peer_gates.get(
                    family_map.get(row.scenario_name, row.scenario_name), 0)
                for row in rows], dtype=np.float32)

        def hook(aid, observations, indices):
            result = observations.new_zeros(())
            raw = gated = anchor_loss = observations.new_zeros(())
            if mutual_enabled:
                gate = torch.as_tensor(sample_gates[aid][indices], device=self.device)
                raw = student_teacher_kl(self.actors[aid], self.peer[aid], observations,
                    direction=direction, reduction=reduction)
                gated = student_teacher_kl(self.actors[aid], self.peer[aid], observations, gate,
                    direction=direction, reduction=reduction)
                result = result + self.beta * gated
            if self.anchor_beta > 0:
                anchor_loss = student_teacher_kl(self.actors[aid], self.anchor[aid], observations)
                result = result + self.anchor_beta * anchor_loss
            n = int(observations.shape[0])
            aux_totals[aid]["n"] += n
            for label, term in (("raw", raw), ("gated", gated), ("anchor", anchor_loss)):
                aux_totals[aid][label] += float(term.detach().item()) * n
            return result

        with self.rng_context():
            # Generate once per rollout, from the checkpointed learner stream.
            # The same weights persist across PPO epochs and minibatch shuffles.
            for aid in BLUE_IDS:
                sample_gates[aid] = (build_distillation_gates(evidence_gates[aid],
                    variant=self.distillation["gate_variant"], rng=np.random)
                    if mutual_enabled and len(evidence_gates[aid]) else evidence_gates[aid].copy())
            if self.algorithm == "ippo":
                role_results = {}
                for aid in BLUE_IDS:
                    if not buffer.data[aid]:
                        if buffer.episode_boundaries[aid] or buffer.episode_bootstrap_states[aid]:
                            raise ValueError("empty IPPO role data has episode/bootstrap metadata")
                        continue
                    role_results[aid] = self._ppo_update(_single_role_buffer(buffer, aid),
                        [aid], {aid: self.actors[aid]}, {aid: self.actor_optims[aid]},
                        self.critics[aid], self.critic_optims[aid], self.value_normalizers[aid],
                        hook if self.anchor_beta > 0 else None)
                if not role_results:
                    raise ValueError("cannot update IPPO from an empty blue team buffer")
                metrics = {"per_role_metrics": {aid: result.metrics for aid, result in role_results.items()}}
                ratio_errors, stop_epochs = {}, {}
                for aid, role_result in role_results.items():
                    ratio_errors.update(role_result.preupdate_ratio_errors)
                    stop_epochs.update(role_result.agent_stop_epoch)
                    for key, value in role_result.metrics.items():
                        # Actor keys already end with the role ID. Critic/stat
                        # keys need an explicit role suffix to avoid pooling.
                        metric_key = key if key.endswith(f"_{aid}") else f"{key}_{aid}"
                        metrics[metric_key] = value
            else:
                result = self._ppo_update(buffer, self.agent_ids, self.actors, self.actor_optims,
                    self.critic, self.critic_optim, self.value_normalizer,
                    hook if mutual_enabled or self.anchor_beta > 0 else None)
                metrics = dict(result.metrics)
                ratio_errors, stop_epochs = result.preupdate_ratio_errors, result.agent_stop_epoch
        self.update_count += 1
        self.schedule_state["update"] = self.update_count
        metrics.update({"team": self.name, "update": self.update_count, "method": self.method,
                        "algorithm": self.algorithm, "critic_input_scope": self.model_spec["critic_input_scope"],
                        "value_normalizer_scope": self.model_spec["value_normalizer_scope"],
                        "distillation": copy.deepcopy(self.distillation),
                        "mutual_beta": self.beta if mutual_enabled else 0.0, "anchor_beta": self.anchor_beta,
                        "peer_hash": self.peer_snapshot["hash"] if self.peer_snapshot else None,
                        "preupdate_ratio_errors": ratio_errors,
                        "agent_stop_epoch": stop_epochs})
        for aid in BLUE_IDS:
            denominator = max(1, aux_totals[aid]["n"])
            metrics.update({f"mutual_kl_{aid}": aux_totals[aid]["raw"] / denominator,
                            f"mutual_gated_kl_{aid}": aux_totals[aid]["gated"] / denominator,
                            f"anchor_kl_{aid}": aux_totals[aid]["anchor"] / denominator,
                            f"gate_fraction_{aid}": float(sample_gates[aid].mean()) if len(sample_gates[aid]) and mutual_enabled else 0.0,
                            f"gate_weight_mean_{aid}": float(sample_gates[aid].mean()) if len(sample_gates[aid]) and mutual_enabled else 0.0,
                            f"gate_nonzero_coverage_{aid}": float((sample_gates[aid] > 0).mean()) if len(sample_gates[aid]) and mutual_enabled else 0.0,
                            f"evidence_gate_fraction_{aid}": float(evidence_gates[aid].mean()) if len(evidence_gates[aid]) and mutual_enabled else 0.0})
        return metrics

    def _ppo_update(self, buffer, ids, actors, actor_optims, critic, critic_optim,
                    value_normalizer, actor_aux_loss=None):
        """Call the same production PPO path for centralized or role-local value."""
        cfg = self.config
        context = f"Mut {self.algorithm.upper()} {self.name} roles={ids}"
        prepared = prepare_mappo_update_batch(buffer, critic, ids, self.device,
            gamma=float(cfg.get("gamma", 0.997)), gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            obs_dim=20, state_dim=80, action_dim=2, context=f"{context} prepare",
            value_normalizer=value_normalizer)
        return run_mappo_ppo_update(prepared, actors, actor_optims, critic, critic_optim, ids,
            obs_dim=20, state_dim=80, action_dim=2,
            ppo_epochs=int(cfg.get("ppo_epochs", 5)), minibatch_size=int(cfg.get("minibatch_size", 256)),
            clip_eps=float(cfg.get("clip_eps", 0.15)), ent_coef=float(self.schedule_state["entropy"]),
            max_grad_norm=float(cfg.get("max_grad_norm", 0.5)), target_kl=float(cfg.get("target_kl", 0.02)),
            value_normalizer=value_normalizer, huber_delta=float(cfg.get("huber_delta", 10)),
            shuffle=bool(cfg.get("shuffle", True)), context=f"{context} update",
            actor_aux_loss=actor_aux_loss)

    def state_dict(self) -> dict:
        return copy.deepcopy({"protocol": LEARNER_PROTOCOL, "name": self.name, "seed": self.seed,
            "config": self.config, "model_spec": self.model_spec, "action_protocol": ACTION_PROTOCOL_VERSION,
            "actors": self.export_actors(),
            "critic": ({aid: self.critics[aid].state_dict() for aid in BLUE_IDS}
                if self.algorithm == "ippo" else self.critic.state_dict()),
            "actor_optims": {aid: opt.state_dict() for aid, opt in self.actor_optims.items()},
            "critic_optim": ({aid: self.critic_optims[aid].state_dict() for aid in BLUE_IDS}
                if self.algorithm == "ippo" else self.critic_optim.state_dict()),
            "value_normalizer": ({aid: self.value_normalizers[aid].state_dict() for aid in BLUE_IDS}
                if self.algorithm == "ippo" else self.value_normalizer.state_dict()),
            "rng": capture_rng_state() if self._in_rng_context else self._rng_state,
            "scenario_rng": self.scenario_rng.getstate(), "update_count": self.update_count,
            "cost_state": self.cost_state, "schedule_state": self.schedule_state,
            "anchor": {aid: actor.state_dict() for aid, actor in self.anchor.items()},
            "peer_snapshot": self.peer_snapshot, "peer_gates": self.peer_gates})

    def load_state_dict(self, state: Mapping) -> None:
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("learner resume schema fields differ")
        for field in ("protocol", "name", "seed", "config", "model_spec", "action_protocol"):
            matches = self._compatible_model_spec(state[field]) if field == "model_spec" else state[field] == expected[field]
            if not matches:
                raise ValueError(f"learner resume identity differs: {field}")
        if state["update_count"] < 0 or state["schedule_state"]["update"] != state["update_count"]:
            raise ValueError("learner update/schedule counters disagree")
        if self.algorithm == "ippo":
            for field in ("critic", "critic_optim", "value_normalizer"):
                if set(state[field]) != set(BLUE_IDS):
                    raise ValueError(f"independent IPPO resume {field} roles differ")
        for aid in BLUE_IDS:
            self.actors[aid].load_state_dict(state["actors"][aid], strict=True)
            self.actor_optims[aid].load_state_dict(state["actor_optims"][aid])
            self.anchor[aid].load_state_dict(state["anchor"][aid], strict=True)
        if self.algorithm == "ippo":
            for aid in BLUE_IDS:
                self.critics[aid].load_state_dict(state["critic"][aid], strict=True)
                self.critic_optims[aid].load_state_dict(state["critic_optim"][aid])
                self.value_normalizers[aid].load_state_dict(state["value_normalizer"][aid], strict=True)
                self.value_normalizers[aid].validate_state()
        else:
            self.critic.load_state_dict(state["critic"], strict=True)
            self.critic_optim.load_state_dict(state["critic_optim"])
            self.value_normalizer.load_state_dict(state["value_normalizer"], strict=True)
            self.value_normalizer.validate_state()
        self._rng_state = copy.deepcopy(state["rng"])
        self.scenario_rng.setstate(state["scenario_rng"])
        self.update_count = int(state["update_count"])
        self.cost_state, self.schedule_state = copy.deepcopy(state["cost_state"]), copy.deepcopy(state["schedule_state"])
        self.set_peer(state["peer_snapshot"], state["peer_gates"])
