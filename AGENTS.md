# Project conventions

- Keep changes minimal. Do not add auxiliary scripts, reports or other files
  unless needed for the requested functionality or explicitly requested.
- Use the existing `dreamwaq` Conda environment for IsaacLab. Start local commands
  with `bash scripts/isaac.sh` so activation and library paths are applied.
- Formal training is started manually by the user. Do not automatically start or
  stop it as part of implementation or verification.
- All IsaacLab and future Newton training artifacts must share the project root
  `logs/rsl_rl/`, with separate backend subdirectories:
  `logs/rsl_rl/isaaclab/<experiment>/<run>/` and
  `logs/rsl_rl/newton/<experiment>/<run>/`.
- Keep each run's checkpoints, TensorBoard events and resolved configuration
  together. Never mix backend checkpoints in one run or overwrite old results.
- Point comparison TensorBoard at `logs/rsl_rl/` to include both backends. Future
  Newton integration must use this convention rather than its third-party
  package's default runs directory.
- Preserve algorithm and task settings when changing output paths. Keep backend,
  task, seed and training progress identifiable when comparing results.
