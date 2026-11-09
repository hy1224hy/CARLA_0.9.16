# CARLA reward experiment archive — 2026-11-09

The user ended the experiment. This archive records R6 at its last verified complete
checkpoint, **u90/100**, not a completed u100 run. No Python or CARLA process was
present during inspection. Historical state/worker/session files still say RUNNING;
they are retained as original evidence and do not establish a live process.
The exit cause and whether the completion hook ran are unconfirmed.

The compressed JSONL files preserve the original logs byte for byte, including any
partial u91 records. Filter by checkpoint round and matching committed segment
before analysis. `r6_u90.pt` is the exact manifest-verified complete checkpoint.
The manifest's checkpoint path refers to the original run directory; in this
archive the same bytes are stored as `r6_u90.pt`.

Training used Town05, B0, seed42, a 42-dimensional two-blue team critic, and R6
single-blue policy-replacement training samples. Independent evaluations were
cancelled. A/B are not independent seeds. Training rewards alone do not establish
successful adversarial behavior or generalization.

This commit captures code, configuration, documents and compact experiment records.
Large historical raw logs, repeated intermediate weights, temporary files and old
binary bundles remain on local disk. `LOCAL_ARTIFACTS_NOT_ADDED.json` inventories
the untracked files left outside the commit; no such files were deleted.
