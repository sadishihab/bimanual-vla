#!/usr/bin/env python3
"""Transcribe an audio file with the Speechmatics batch transcription API.

    export SPEECHMATICS_API_KEY=...
    python speechmatics/transcribe.py speechmatics_test.mp4

Talks to the real batch REST API, not a client library, so the request shapes
below are worth being explicit about and are cited against source rather than
guessed:

* Base URL ``https://asr.api.speechmatics.com/v2`` is ``BATCH_SELF_SERVICE_URL``
  in the official Python SDK (``speechmatics/constants.py``) -- the self-service
  URL for a normal API key.  ``eu1``/``eu2``/``us2`` regional hosts exist too, but
  those are enterprise high-availability endpoints, not the default, and jobs
  created there do not show up in the Portal.
* Job submission is ``POST /v2/jobs``, ``multipart/form-data`` with two parts: a
  ``config`` part holding the job config as a *JSON string* (not a JSON body --
  the whole request is multipart), and a ``data_file`` part with the raw audio.
* Status is ``GET /v2/jobs/{id}``, ``running -> done | rejected``.
* The transcript is a separate call, ``GET /v2/jobs/{id}/transcript``, not part of
  the status response.  ``format=txt`` returns plain text; the default
  ``json-v2`` returns the full per-word structure this module also exposes,
  because a caller matching against short task strings mostly wants the plain
  text but the word-level detail is there if something needs re-checking.

Authentication is ``Authorization: Bearer $SPEECHMATICS_API_KEY`` -- read from the
environment on every call, never written to disk or logged, and never given a
default, so a missing key fails immediately rather than transcribing as some
other account or silently no-op-ing.

Uses only the standard library.  This integration has one job and does not
warrant a dependency; the multipart body is three parts and is easier to get
right by hand than to trust to a library version we have not pinned.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional

BASE_URL = "https://asr.api.speechmatics.com/v2"
POLL_INTERVAL_S = 3.0
DEFAULT_TIMEOUT_S = 300.0


class SpeechmaticsError(RuntimeError):
    """Anything the API itself objected to -- a 4xx/5xx with its body attached."""


def _api_key() -> str:
    key = os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise SpeechmaticsError(
            "SPEECHMATICS_API_KEY is not set. Export it before running this -- "
            "the key is never hardcoded here and never has a fallback.")
    return key


def _multipart_body(fields: Dict[str, str], file_field: str, file_path: pathlib.Path
                    ) -> tuple[bytes, str]:
    """Encode a multipart/form-data body by hand: a few string parts and one file."""
    boundary = f"----speechmatics-{uuid.uuid4().hex}"
    nl = "\r\n"
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}{nl}"
            f'Content-Disposition: form-data; name="{name}"{nl}{nl}'
            f"{value}{nl}".encode("utf-8"))
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    header = (
        f"--{boundary}{nl}"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"{nl}'
        f"Content-Type: {content_type}{nl}{nl}").encode("utf-8")
    body = b"".join(parts) + header + file_path.read_bytes() + f"{nl}--{boundary}--{nl}".encode()
    return body, boundary


def _request(method: str, path: str, *, api_key: str,
            body: Optional[bytes] = None, content_type: Optional[str] = None,
            query: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    if query:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(query)}"
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:                          # noqa: PERF203
        detail = exc.read().decode("utf-8", "replace")
        raise SpeechmaticsError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else {}


def submit_job(audio_path: pathlib.Path, *, language: str = "en",
              operating_point: str = "enhanced", api_key: Optional[str] = None) -> str:
    """POST /v2/jobs.  Returns the job id."""
    api_key = api_key or _api_key()
    config = {
        "type": "transcription",
        "transcription_config": {"language": language, "operating_point": operating_point},
    }
    body, boundary = _multipart_body({"config": json.dumps(config)}, "data_file", audio_path)
    resp = _request("POST", "jobs", api_key=api_key, body=body,
                    content_type=f"multipart/form-data; boundary={boundary}")
    job_id = resp.get("id")
    if not job_id:
        raise SpeechmaticsError(f"job submission returned no id: {resp}")
    return job_id


def wait_for_job(job_id: str, *, api_key: Optional[str] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Poll GET /v2/jobs/{id} until the job leaves 'running'.  Returns the final status."""
    api_key = api_key or _api_key()
    deadline = time.time() + timeout_s
    while True:
        resp = _request("GET", f"jobs/{job_id}", api_key=api_key)
        status = resp.get("job", {}).get("status", "unknown")
        if status != "running":
            return status
        if time.time() > deadline:
            raise SpeechmaticsError(
                f"job {job_id} still running after {timeout_s:.0f}s; giving up")
        time.sleep(POLL_INTERVAL_S)


def get_transcript(job_id: str, *, fmt: str = "json-v2",
                   api_key: Optional[str] = None) -> Dict[str, Any] | str:
    """GET /v2/jobs/{id}/transcript.  ``fmt='txt'`` returns a plain string."""
    api_key = api_key or _api_key()
    if fmt == "txt":
        url = f"{BASE_URL}/jobs/{job_id}/transcript?format=txt"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SpeechmaticsError(f"GET transcript -> HTTP {exc.code}: {detail}") from exc
    return _request("GET", f"jobs/{job_id}/transcript", api_key=api_key, query={"format": fmt})


def plain_text(transcript_json: Dict[str, Any]) -> str:
    """The words of a json-v2 transcript, joined the way a reader would say them.

    Punctuation tokens attach to the previous word with no leading space; everything
    else gets one.  Simple and exact for this scene's short single-sentence commands.
    """
    out = []
    for item in transcript_json.get("results", []):
        alt = (item.get("alternatives") or [{}])[0]
        word = alt.get("content", "")
        if item.get("type") == "punctuation" and out:
            out[-1] += word
        else:
            out.append(word)
    return " ".join(out)


def transcribe(audio_path: pathlib.Path, *, language: str = "en",
              operating_point: str = "enhanced", timeout_s: float = DEFAULT_TIMEOUT_S
              ) -> Dict[str, Any]:
    """Submit, wait, and fetch: the whole batch round trip for one file.

    Returns both the plain text and the full json-v2 payload, and the job id, so a
    caller can inspect per-word confidence without a second network call.
    """
    api_key = _api_key()
    job_id = submit_job(audio_path, language=language, operating_point=operating_point,
                        api_key=api_key)
    status = wait_for_job(job_id, api_key=api_key, timeout_s=timeout_s)
    if status != "done":
        raise SpeechmaticsError(f"job {job_id} finished with status {status!r}, not 'done'")
    transcript_json = get_transcript(job_id, fmt="json-v2", api_key=api_key)
    return {"job_id": job_id, "status": status,
            "text": plain_text(transcript_json), "transcript": transcript_json}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=pathlib.Path, help="path to an audio file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--operating-point", default="enhanced",
                        choices=("standard", "enhanced"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--json", action="store_true", help="print the full json-v2 payload")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"no such file: {args.audio}")

    result = transcribe(args.audio, language=args.language,
                        operating_point=args.operating_point, timeout_s=args.timeout)
    print(f"job {result['job_id']}: {result['status']}")
    print(f"transcript: {result['text']!r}")
    if args.json:
        print(json.dumps(result["transcript"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except SpeechmaticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
