# Dangerous interaction evaluation implementation plan

Goal: evaluate existing frozen red / trained blue checkpoints without changing training or task identity.
Architecture: pure OBB geometry/time metrics; scoped evaluation-only geometry capture; independent postprocessor and batch CLI; versioned protocol and provenance.

- [x] Inspect trace lifecycle, checkpoint validation, missing bbox, paired retry semantics.
- [x] Test and implement constant-velocity fixed-yaw OBB TTC, TET/TIT, persistent events and observed responses.
- [x] Test and implement same-case/team/seed/attempt pairing, raw and qualified yield decomposition.
- [x] Add live wrapper and offline trace analysis with actual bbox capture, no physics ticks or training-source edits.
- [x] Freeze configurable diagnostic protocol and distinguish literature thresholds from project choices.
- [x] Run geometry, pairing, capture and existing evaluation regression tests; review independently.
- [x] Document server commands; commit and push only monitor branch files.

No reference reliability or collision fault is inferred from one outcome. Missing data stays missing. A/B and repeated evaluations are dependent. Stage and identities stay explicit. Existing summary-only reports lack geometry; rerun evaluation, not training.
