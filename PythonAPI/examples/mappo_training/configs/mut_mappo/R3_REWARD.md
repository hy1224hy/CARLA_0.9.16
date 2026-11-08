# R3 individual rewards

Profile: `individual_driving_critical_v3`. Opt-in; R2 unchanged.

For each active blue i:

    r_i = alpha * sum(frozen_v6_soft_i) + sum(frozen_v6_hard_i)
          - task_violation_cost * first_task_violation_i
          + beta * (critical_i + primary_event_bonus_i + severe_i)

Defaults alpha=0.3, beta=1. Hard components are collision, offroad, road_rule;
their signed original v6 values are retained once, without scaling. All other
registered v6 components are soft, including progress, success, timeout, lane,
pace, proximity and smoothness. Missing registered components fail explicitly.
No R2 extra normal-driving penalties, C1 progress or completed-task tail objective.

Each blue has an independent geometry tracker comparing only itself against the
two red vehicles. Exposure, cap, bonus and failure eligibility are per-blue.
Another blue neither earns that reward nor suppresses its eligibility. Per-role
exposure cap is 2 seconds; critical positive reward cap is 3; event bonus is 5
once, after 1 second grace and a same-pair TTC 0.5--1.5s lasting 0.3s. Severe
TTC retains negative shaping. Event payment is immediate and is NOT retracted
on a later failure, matching R2 timing; final-qualified event metrics are separate.

Actors/agent-conditioned centralized critic/PPO/GAE are unchanged. Each actor
uses its own returns. No team tail is inserted for R3 after a vehicle departs.
Individual components and adversarial metrics are logged; team reward components
and utility are per-role sums divided by two for reporting only. Team-wide geometry
metrics are still measured independently for evaluation and never assigned to actors.

Alpha is unnormalized scaling, not calibrated or guaranteed effective. Validate
component scales and scenario outcomes in a pilot before long training. Positive
budget validation is not a safety guarantee.

r3_individual_candidate_v1.json is a local template, not a server-ready long run.
Validate checkpoint, case manifest, prepared reference path and budgets first.
References remain required by the existing evaluation protocol, although absent
from the R3 objective. No live CARLA training was launched for this change.
