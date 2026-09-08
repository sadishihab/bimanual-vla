# Working rules for this repo

## Commits

- Commit after each completed task, without being asked.
- Never commit `.venv/`, `.cache/`, `third_party/`, or generated images.

## Test scripts

- **One seed per process.** No multi-seed loops inside a single process — to
  sweep seeds, invoke the script once per seed from the shell.
- No video, and no renderer unless explicitly asked for. Memory has been a
  problem in this repo; a script that only needs to check physics or IK must
  not construct a `mujoco.Renderer` at all. If one is genuinely needed, free it.

## Reporting failures

- When something fails, report the mechanism and the measurement — the contact
  normals, the residual, the clearance in millimetres — not just that it failed.
- Do not change the scene or the task to manufacture a success. A primitive that
  fails against honest geometry is a finding, not a bug to tune away.
