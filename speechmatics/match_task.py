#!/usr/bin/env python3
"""Match a transcript to the closest of the task-conditioned policy's seven strings.

    python speechmatics/match_task.py "place the fork in the table setting"

Only seven candidates and short sentences, so this does not warrant an embedding
model or a second network call: :mod:`difflib`'s ``SequenceMatcher`` ratio, run on
lower-cased text with punctuation stripped, is exact enough and has no dependency
beyond the standard library.  The seven strings are also exactly the set the
recorded dataset's task table was built from -- see ``control/language.py`` and
``checkpoints/*/task_embeddings.json`` -- so a matched string here is guaranteed to
be one the checkpoint has an embedding for, with no risk of asking it to embed
something novel at inference time.
"""

from __future__ import annotations

import argparse
import difflib
import re
from typing import Dict, List, Sequence, Tuple

TASKS: Tuple[str, ...] = (
    "Hand the fork to the other arm and place it in the table setting",
    "Hand the mug to the other arm and place it in the table setting",
    "Hand the spoon to the other arm and place it in the table setting",
    "Place the fork in the table setting",
    "Place the mug in the table setting",
    "Place the plate in the table setting",
    "Place the spoon in the table setting",
)

# A margin the best match has to clear over the runner-up before it is trusted.
# The seven strings are close to each other by design (four share "in the table
# setting", three share "Hand the ... to the other arm"), so a ratio alone can be
# high even for the wrong one; the gap to whatever is second is what actually says
# the transcript picked out one prop over another.
AMBIGUOUS_MARGIN = 0.05


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)          # drop punctuation the ASR may add
    return re.sub(r"\s+", " ", text)


def score_all(transcript: str, tasks: Sequence[str] = TASKS) -> List[Tuple[str, float]]:
    """Every candidate with its similarity ratio, best first."""
    norm = _normalize(transcript)
    scored = [(t, difflib.SequenceMatcher(None, norm, _normalize(t)).ratio()) for t in tasks]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def match_task(transcript: str, tasks: Sequence[str] = TASKS) -> Dict[str, object]:
    """The best match, its score, the runner-up, and whether the match is confident.

    ``confidence`` is the winner's own ratio -- how close the transcript is to the
    string it matched.  ``ambiguous`` is a separate question: whether the runner-up
    was close enough behind that the choice between them was not clear-cut, which a
    high confidence score alone does not rule out when several candidates share most
    of their words.
    """
    ranked = score_all(transcript, tasks)
    best_task, best_score = ranked[0]
    runner_task, runner_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return {
        "transcript": transcript,
        "matched_task": best_task,
        "confidence": best_score,
        "runner_up": runner_task,
        "runner_up_confidence": runner_score,
        "margin": best_score - runner_score,
        "ambiguous": (best_score - runner_score) < AMBIGUOUS_MARGIN,
        "ranked": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", help="the ASR transcript text to match")
    parser.add_argument("--all", action="store_true", help="show every candidate's score")
    args = parser.parse_args()

    result = match_task(args.transcript)
    print(f"transcript : {result['transcript']!r}")
    print(f"matched    : {result['matched_task']!r}")
    print(f"confidence : {result['confidence']:.3f}"
          f"  (runner-up {result['runner_up_confidence']:.3f}, "
          f"margin {result['margin']:.3f}{', AMBIGUOUS' if result['ambiguous'] else ''})")
    if args.all:
        print("\nall candidates:")
        for task, score in result["ranked"]:
            print(f"  {score:.3f}  {task}")


if __name__ == "__main__":
    main()
