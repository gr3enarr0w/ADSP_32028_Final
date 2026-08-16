#!/usr/bin/env python3
"""
run_end_to_end.py — headless end-to-end harness for the full voice turn.

`eval/run_eval.py` measures *retrieval* quality (recall@k / MRR over
`eval/gold_queries.jsonl`). This script is the other axis: it exercises the
whole assembled pipeline — audio → ASR → Router → Planner → Retriever
(private + live) → Answerer/Critic → TTS — and asserts the structural
invariants the assignment grades:

  * every turn produces a spoken `speech` string,
  * that string fits the ≤15s budget (`prompts/answerer_critic.md`),
  * the Critic returns a verdict and nothing is spoken before it accepts,
  * every comparison-table row traces back to a `doc_id` that also appears in
    the citations (the grounding contract in `prompts/answerer_critic.md`),
  * an audio artifact exists whenever synthesis was requested and accepted.

It drives `ui/pipeline.run_turn()` — the same entry point `ui/app.py` uses —
so a green run here is evidence about the demo, not about a parallel code path.

Usage:

    # transcripts from eval/gold_queries.jsonl (no ASR, no audio) — fastest
    PYTHONPATH=src python3 scripts/run_end_to_end.py

    # the real voice path: the ten prerecorded clips in audio/
    PYTHONPATH=src python3 scripts/run_end_to_end.py --source audio

    # offline/CI: hash embedder, no reranker, no synthesis
    EMBEDDING_PROVIDER=hash USE_RERANKER=false \
        PYTHONPATH=src python3 scripts/run_end_to_end.py --no-tts

Results are written to `eval/results/end_to_end_<timestamp>.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "ui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline import run_turn  # noqa: E402
from rag.asr import REFERENCE_TRANSCRIPTS, sample_clips  # noqa: E402
from rag.tts import estimate_speech_seconds, fits_budget  # noqa: E402

GOLD = REPO_ROOT / "eval" / "gold_queries.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def _load_gold_cases() -> list[dict]:
    cases = []
    with GOLD.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                cases.append({"id": row["id"], "transcript": row["query"],
                              "audio_path": None})
    return cases


def _load_audio_cases() -> list[dict]:
    return [
        {"id": clip.stem, "transcript": None, "audio_path": clip,
         "reference": REFERENCE_TRANSCRIPTS.get(clip.name)}
        for clip in sample_clips()
    ]


def _check(trace, final_state, synthesize: bool) -> list[str]:
    """Structural invariants. Returns a list of failure strings (empty = pass)."""
    failures: list[str] = []

    answerer = final_state.get("answerer_output") or {}
    critic = final_state.get("critic_output") or {}
    speech = answerer.get("speech") or ""

    if not speech:
        failures.append("no speech produced")
    elif not fits_budget(speech):
        failures.append(
            f"speech exceeds the 15s budget "
            f"(~{estimate_speech_seconds(speech):.1f}s, {len(speech.split())} words)"
        )

    if critic.get("action") not in {"accept", "revise"}:
        failures.append(f"critic action missing/invalid: {critic.get('action')!r}")

    # Grounding: every table row's doc_id must be cited.
    cited_docs = {c.get("doc_id") for c in (trace.citations or []) if c.get("doc_id")}
    for row in answerer.get("comparison_table") or []:
        doc_id = row.get("doc_id")
        if doc_id and doc_id not in cited_docs:
            failures.append(f"comparison row doc_id={doc_id} is not cited")

    # Nothing spoken before the Critic accepts.
    if trace.audio_path and critic.get("action") != "accept":
        failures.append("audio synthesized despite a non-accepting Critic verdict")

    if synthesize and critic.get("action") == "accept" and speech:
        if not trace.audio_path:
            failures.append("no audio artifact for an accepted answer")
        elif not Path(trace.audio_path).exists():
            failures.append(f"audio path does not exist: {trace.audio_path}")

    if not trace.steps:
        failures.append("empty step log")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["gold", "audio"], default="gold",
                    help="gold = transcripts from eval/gold_queries.jsonl (default); "
                         "audio = the prerecorded clips in audio/ (exercises ASR)")
    ap.add_argument("--no-tts", action="store_true", help="skip speech synthesis")
    ap.add_argument("--allow-asr-fallback", action="store_true",
                    help="permit reference transcripts when faster-whisper is absent")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    args = ap.parse_args()

    cases = _load_gold_cases() if args.source == "gold" else _load_audio_cases()
    if not cases:
        print(f"No cases found for --source {args.source}", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]

    synthesize = not args.no_tts
    started = datetime.now(timezone.utc)
    records, n_failed = [], 0

    print(f"▶ end-to-end: {len(cases)} case(s) from '{args.source}', "
          f"tts={'on' if synthesize else 'off'}\n")

    for case in cases:
        t0 = time.perf_counter()
        record = {"id": case["id"], "source": args.source}
        try:
            trace, final_state = run_turn(
                transcript=case.get("transcript"),
                audio_path=case.get("audio_path"),
                synthesize=synthesize,
                allow_asr_fallback=args.allow_asr_fallback or None,
            )
            failures = _check(trace, final_state, synthesize)
            critic = final_state.get("critic_output") or {}
            record.update({
                "transcript": trace.transcript,
                "reference": case.get("reference"),
                "speech": trace.answer_text,
                "speech_seconds_est": round(estimate_speech_seconds(trace.answer_text), 1),
                "critic_action": critic.get("action"),
                "grounded": critic.get("grounded"),
                "n_steps": len(trace.steps),
                "nodes": [s.node for s in trace.steps],
                "n_citations": len(trace.citations),
                "n_table_rows": len((final_state.get("answerer_output") or {})
                                    .get("comparison_table") or []),
                "discrepancies": [
                    {"title": i.get("title"), **i["discrepancy"]}
                    for i in (final_state.get("reconciled") or {}).get("items", [])
                    if i.get("discrepancy")
                ],
                "audio_path": trace.audio_path,
                "wall_ms": round((time.perf_counter() - t0) * 1000),
                "failures": failures,
                "ok": not failures,
            })
        except Exception as e:  # noqa: BLE001 - one bad case shouldn't abort the sweep
            record.update({"ok": False, "failures": [f"exception: {e!r}"],
                           "wall_ms": round((time.perf_counter() - t0) * 1000)})

        if not record["ok"]:
            n_failed += 1
        icon = "✔" if record["ok"] else "✘"
        print(f"  {icon} {record['id']}  ({record.get('wall_ms', 0)} ms)  "
              f"{(record.get('transcript') or '')[:60]!r}")
        for f in record.get("failures", []):
            print(f"      ! {f}")
        records.append(record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"end_to_end_{stamp}.json"
    summary = {
        "started_utc": started.isoformat(),
        "source": args.source,
        "synthesize": synthesize,
        "n_cases": len(records),
        "n_passed": len(records) - n_failed,
        "n_failed": n_failed,
        "mean_wall_ms": round(sum(r.get("wall_ms", 0) for r in records) / max(len(records), 1)),
        "cases": records,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n{summary['n_passed']}/{summary['n_cases']} passed  ·  "
          f"mean {summary['mean_wall_ms']} ms/turn")
    print(f"→ {out_path.relative_to(REPO_ROOT)}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
