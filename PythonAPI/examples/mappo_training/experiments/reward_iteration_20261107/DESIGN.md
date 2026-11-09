# Overnight reward iteration, 2026-11-07

Authorized: each new version trains fresh for 66 updates. B0, seed42, same frozen red source and case list. Individual driving and individual adversarial rewards; centralized critic unchanged. User prioritizes dangerous exposure and permits higher bad rate. No collision-rate threshold will be silently presented as approved.

Compare deterministic fixed selection cases, per role and family. Main score is valid pre-contact bounded TTC exposure; also report qualified exposure, event rate, collisions/offroad/rules and task completion. More collisions alone is not success. Team-wide geometry is diagnostic; per-role geometry/reward is the training objective. Preserve test split for one final frozen candidate.

Sequence: frozen base-policy u0 evaluation; new alpha0.1 66-update experiment; analyze before selecting next change. Candidate next interventions: alpha0.03 if driving still dominates with available attack opportunities; separate progress/success from soft costs if safety costs vanish; bounded approach shaping only when local events remain sparse despite feasible opportunities. No edits to a running worker's imported reward code. Every version retains config, rationale, source hashes/snapshot, metrics, checkpoint, deterministic evaluation and verdict. Training repeats always use fresh source actors/optimizers/critic, not previous adversarial checkpoint.

Monitoring: read-only 20-minute window analyses and 5-hour durable progress summaries; neither guarantees API quota availability. Preserve a resumable state file and stop on CARLA/worker errors instead of blind restart.

Archived R3 u66 has LR horizon325; it is historical comparison only. Primary baseline is u0 with verified source actors. New horizon66 introduces a confound. Use a matching fresh alpha0.3/horizon66 control for causal claims; archived baseline comparison is exploratory.
