# Uploaded models

Place frozen external checkpoints here, for example `red_weak_u370.pt`.
Weights and their checksum sidecars are ignored by Git. Keep model files
immutable during training. Use `utils.prepare_mut_mappo_server` with the
SHA256 computed on the source machine to verify and register the upload.

Code/configuration templates belong to Git; models, generated configurations,
logs and evaluation outputs remain machine-local. See
`docs/server_mut_mappo_usage.md` for the workflow in the existing checkout.
