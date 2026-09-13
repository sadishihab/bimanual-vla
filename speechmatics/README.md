# Voice command integration

Three isolated pieces, none of which touch `control/`, `envs/`, or the existing
eval scripts:

| file | does |
|---|---|
| `transcribe.py` | audio file → text, via the Speechmatics batch REST API |
| `match_task.py` | text → the closest of the checkpoint's 7 known task strings |
| `scripts/voice_command_demo.py` | wires both into `scripts/eval_policy.py`'s own directed-episode machinery |

## Speechmatics API — cited, not guessed

`transcribe.py` talks to the real batch REST API with the standard library only
(`urllib`, hand-built multipart), against endpoints and a base URL confirmed
directly from the official Python SDK source
(`speechmatics/constants.py`, `BATCH_SELF_SERVICE_URL`) and the published OpenAPI
spec (`docs.speechmatics.com/batch.yaml`), not from marketing pages:

- base URL: `https://asr.api.speechmatics.com/v2` — the self-service SaaS default.
  (`eu1`/`eu2`/`us2` regional hosts exist but are enterprise HA endpoints; jobs made
  there don't appear in the Portal.)
- `POST /v2/jobs` — `multipart/form-data`, two parts: `config` (a JSON *string*,
  not a JSON body — the whole request is multipart) and `data_file` (the raw audio).
- `GET /v2/jobs/{id}` — poll until `status` leaves `"running"`.
- `GET /v2/jobs/{id}/transcript?format=json-v2|txt` — the transcript is a
  **separate call**, not part of the status response.
- auth: `Authorization: Bearer $SPEECHMATICS_API_KEY`, read from the environment
  on every call. Never hardcoded, never given a default — a missing key fails
  immediately (`SpeechmaticsError`, exit 1) rather than transcribing under some
  other account or silently doing nothing. Verified: unsetting the key gives a
  clean one-line error, not a traceback.

## Matching

Only 7 candidates and short sentences, so `match_task.py` uses `difflib` — no
embedding model, no second network call. It reports both the winner's confidence
and the **margin** to the runner-up, because the seven strings are close to each
other by design (four share "in the table setting", three share "to the other
arm"), so a high ratio alone doesn't mean the choice between props was clear-cut.
An `AMBIGUOUS` flag fires when the margin is under 0.05.

Verified against the exact target phrase and several adversarial ones:

```
"place the fork in the table setting"        -> Place the fork ...    conf 1.000 margin 0.099
"hand the fork ... to the other arm ..."      -> Hand the fork ...     conf 1.000 margin 0.054
"give the fork to the other arm"              -> Place the fork ...    conf 0.585 margin 0.031  AMBIGUOUS
```

The seven candidates are exactly the set `control/language.py`'s task table was
built from, so a matched string is always one a conditioned checkpoint has a
real embedding for.

## The demo script

`scripts/voice_command_demo.py` imports `scripts/eval_policy.py`'s `run_episode`,
`fresh_state`, `make_sanity` and friends directly rather than re-implementing the
closed-loop mechanics — so a voice-driven episode runs the identical code path the
directed evaluation numbers in `README.md` were measured on, not a parallel one
that might quietly diverge. It defaults to `checkpoints/step_0044000_shuffled`
(the checkpoint whose de-confounded training data is what gave language
conditioning a measurable effect — see the top-level README).

```sh
export SPEECHMATICS_API_KEY=...
MUJOCO_GL=egl <lerobot-env>/bin/python scripts/voice_command_demo.py \
    --audio speechmatics_test.mp4 --seed 0
```

Needs an environment with mujoco *and* lerobot, same as `eval_policy.py` — the
project's own `.venv` only has mujoco. The observation sanity check from
`eval_policy.py` carries over unchanged: a constant-image render aborts the run
before any control step, rather than letting the policy act blind.

## Not yet run against the live API

Everything up to the network call is verified in this repo: the multipart body is
byte-exact (boundary markers, the raw audio present verbatim, the config JSON
intact), the missing-key path fails cleanly, and the matcher is confirmed correct
on the literal target sentence and several near-misses. The one thing that needs
`SPEECHMATICS_API_KEY` set is the actual transcription call against
`speechmatics_test.mp4` — that number is reported once the key is available.
