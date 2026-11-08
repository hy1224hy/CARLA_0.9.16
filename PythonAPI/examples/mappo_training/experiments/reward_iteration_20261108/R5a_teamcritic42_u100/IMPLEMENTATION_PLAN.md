# R5a team critic repair

Approved scope: keep R5a rewards, B0, seed42, Town05 and 100 updates unchanged. Each A/B team inherits original blue actors and a separate copy of the base two-agent critic plus ValueNorm. Fresh optimizers; no critic-only warmup; no live CARLA calls until the user starts the server. Preserve old runs. Pre-change commit: 44b1e53.

Implementation:
- [x] Regression checks: predictions match the original critic for blue observations with physical-ID mapping, ignore red slots, teams own separate storage, joint PPO updates actor/critic, resume restores new critic and rejects old topology.
- [x] Add a two-blue observation adapter retaining the existing 80-value rollout transport while the actual value MLP receives 40 blue values plus 2 identity values. Preserve base net.* parameter keys.
- [x] Add explicit critic_scope/critic_initialization settings; keep existing 84-input legacy mode available for old configurations; strictly load critic and ValueNorm from the verified base checkpoint.
- [x] Wire trainer initialization and source hashes; keep actor-only KL auxiliary path unchanged. Persist actual critic dimensions in each learner model_spec so incompatible resume/peer topology is rejected.
- [x] Prepare a fresh R5a_teamcritic42_u100 directory with unchanged reward config, rationale, source identity and a manual module launch command. Do not touch failed R5a state or start CARLA.
- [x] Run only focused offline regression/config checks, verify real checkpoint tensor and value equality, document results, then ask user to start CARLA.

Important limit: this experiment changes critic observation scope AND initialization, including the accompanying ValueNorm statistics; it does not isolate their individual effects or guarantee convergence. Four physical cars remain in each simulation, A/B run separately, B0 mutual beta=0 remains disabled.

Verification complete: 8 focused tests and actual module --validate-only passed; 274 source snapshot files match live files. No simulator connection or training started.
