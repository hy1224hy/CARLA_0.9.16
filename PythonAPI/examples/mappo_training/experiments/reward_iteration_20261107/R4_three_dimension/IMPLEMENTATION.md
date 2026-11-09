# R4 first candidate, implementation record

Each blue receives its own reward:

r_i = .03*(progress+success+mission_timeout+pace)_i
      + .3*(remaining registered v6 soft components)_i
      + collision_i + offroad_i + road_rule_i - 20*first_task_violation_i
      + min(remaining risk budget 8, 1.0*max(TTC score,spacing score)*dt)
      + min(remaining response budget 2, .5*positive response excess*dt)
      - 2*severe_indicator*dt.

No one-time event bonus. Actor/critic/observation/action/PPO unchanged. First candidate keeps collision=-50; collision_scale=.5 exists for a separate -25 ablation but is not enabled. Hard cost scaling and positive budget are explicitly validated; this is not a safety proof.

Diagnostics are uncapped real durations and longest SAME-PAIR runs. Reward has 1-second warmup, diagnostics do not. Positive credit stops after known contact, OBB contact or own safety failure; another blue's failure does not assign its penalty to this blue. Any severe pair suppresses positive reward for that blue at the current step. Own collision can be reported independently from red-only contact. Post-contact response excluded.

Spacing is SUPPORT-PROJECTED vehicle-boundary clearance, not exact Euclidean polygon distance. Aligned within 30 degrees: front risk requires lateral overlap, positive front gap; dynamic threshold d=2+v_red+closing_speed^2/6. Side risk requires longitudinal overlap and positive side clearance below .75+.05*v_red. These are explicit engineering defaults, not claimed empirically calibrated or legal safety limits. Low-speed headway (<.5m/s) is unavailable. Crossing conflicts remain TTC-based. Scores max-combine TTC and spacing to avoid adding the same geometric risk twice.

Red response proxy=.5*clip(deceleration/3)+.25*clip(abs(longitudinal jerk)/10)+.25*clip(steer rate/1). Finite differences are unfiltered and may be noisy; metrics preserve raw max deceleration and hard-brake duration. Reward uses positive excess over same-case/same-step frozen base-policy red response, gated by this blue's recent risk (1s lag). Missing reference steps produce no response bonus and coverage is reported, never extrapolated. This is temporal association, not causal blame. Lane deviation is not included without road-reference measurement; derivative transverse acceleration is deferred rather than falsely reported.

References are generated with C2 frozen INITIAL blue actors (not learned adversaries or lane PID). They are fixed per case seed, stored/checksummed in reference manifest, included in checkpoint identity. Training seeds vary, so response differences are diagnostic residuals with state/time alignment confounds. Offline retrospective reference uses saved base_u0 traces. Paired final evaluation must also compare base_u0 under the new protocol; old capped exposure/qualification is legacy diagnostic only and must not determine R4 success.

Validation: same-speed short gap without TTC, rear/far non-credit, durations beyond 2s, same-pair continuity, contact blocking, missing geometry, role-specific rewards, fixed-reference residual zero and explicit collision scaling. Synthetic training-chain test covers config/identity/PPO ABI. No claim of improved behavior before actual training and fixed evaluation.
