"""Generate notebooks/03_tts_summary.ipynb (Final deliverable) programmatically."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_tts_summary.ipynb"

nb = new_notebook()
cells = []

# 1. Title / overview
cells.append(new_markdown_cell(
    "# CP-final — Text-to-Speech: ≤15s Spoken Summaries with On-Screen Citations\n"
    "\n"
    "**Owner:** Shane · **Deliverable:** Final — text-to-speech synthesis\n"
    "\n"
    "This notebook turns the Answerer's structured payload (`speech`, "
    "`citations`, `comparison_table` — see `prompts/answerer_critic.md`) into "
    "a **spoken audio file**, using `src/rag/tts.py`:\n"
    "\n"
    "* `estimate_speech_seconds(text)` / `fits_budget(text)` — enforce the "
    "≤15s (~55-word) spoken-answer budget before synthesis.\n"
    "* `speak(text, out_path=None, provider=None, voice=None)` — fragment-based "
    "synthesis to a finished `.wav`/`.mp3` file, dispatching to `pyttsx3` "
    "(offline default), `openai`, or `elevenlabs`.\n"
    "\n"
    "It runs with **zero setup and zero API keys** — the default provider is "
    "`pyttsx3`, a fully offline system-TTS engine — and degrades gracefully to "
    "that default if a paid provider is requested without its key/package, the "
    "same pattern Clark's `web_search.py` uses for search providers."
))

# 2. Setup header
cells.append(new_markdown_cell("## 0. Setup"))

# 3. Setup code
cells.append(new_code_cell(
    "import os, sys\n"
    "sys.path.append(os.path.abspath('../src'))\n"
    "from IPython.display import Audio\n"
    "\n"
    "from rag.config import get_config\n"
    "from rag import tts\n"
    "\n"
    "cfg = get_config()\n"
    "print('TTS provider :', cfg.tts_provider)\n"
    "print('TTS voice    :', cfg.tts_voice or '(engine default)')\n"
    "print('OPENAI_API_KEY set?     :', bool(os.environ.get('OPENAI_API_KEY')))\n"
    "print('ELEVENLABS_API_KEY set? :', bool(os.environ.get('ELEVENLABS_API_KEY')))"
))

# 4. Why fragment-based
cells.append(new_markdown_cell(
    "## 1. Why fragment-based synthesis\n"
    "\n"
    "There are two ways to turn text into spoken audio:\n"
    "\n"
    "* **Streaming synthesis** — audio is generated and played back "
    "incrementally as tokens/words arrive, so playback can start before the "
    "full text exists.\n"
    "* **Fragment-based synthesis** — the complete text is generated first, "
    "then synthesized to a finished audio file in one call, which is then "
    "played back or handed off as a whole.\n"
    "\n"
    "**This notebook uses fragment-based synthesis.** The Answerer's `speech` "
    "field is not safe to speak until the Critic has verified it (grounding + "
    "safety, `prompts/answerer_critic.md`) — there is no partial prefix of the "
    "text that is safe to say early. Since the whole payload must exist and "
    "pass the Critic gate before anything is spoken, streaming buys nothing "
    "here, while fragment-based synthesis is simpler and works uniformly "
    "across every provider, including the fully offline `pyttsx3` default."
))

# 5. Load sample payload
cells.append(new_markdown_cell(
    "## 2. Load a sample Answerer payload\n"
    "\n"
    "In production, this payload comes straight from the Answerer/Critic node "
    "(Victoria's part) after the Critic returns `action: \"accept\"` — see "
    "`prompts/answerer_critic.md`. Here we load it from the same fixture the "
    "prompt disclosure ships: `prompts/fewshots/answerer_examples.json`."
))

# 6. Load fixture code
cells.append(new_code_cell(
    "import json\n"
    "\n"
    "with open('../prompts/fewshots/answerer_examples.json') as f:\n"
    "    examples = json.load(f)\n"
    "\n"
    "example = examples[0]\n"
    "print('speech:\\n ', example['output']['speech'])\n"
    "print()\n"
    "print('citations:')\n"
    "for c in example['output']['citations']:\n"
    "    print(' -', c['title'], '|', c['doc_id'], '|', c['url'])\n"
    "print()\n"
    "print('comparison_table:')\n"
    "for row in example['output']['comparison_table']:\n"
    "    print(' -', row)"
))

# 7. Budget header
cells.append(new_markdown_cell(
    "## 3. Enforce the ≤15s / ≤~55-word budget\n"
    "\n"
    "`prompts/answerer_critic.md` requires the spoken text to be ≤15 seconds "
    "(~2–3 sentences, ≤~55 words) and the Critic is supposed to reject "
    "anything longer. `tts.speak()` re-checks this defensively anyway — "
    "belt-and-suspenders — so a bug upstream never crashes synthesis, it just "
    "truncates and warns."
))

# 8. Budget code
cells.append(new_code_cell(
    "speech_text = example['output']['speech']\n"
    "too_long_text = (\n"
    "    \"My top pick is the GreenGleam Steel-Safe Eco cleaner, a plant-based \"\n"
    "    \"stainless steel cleaner and polish that costs about $12.49 for a 16 \"\n"
    "    \"ounce bottle and has an average rating of 4.6 out of 5 stars from \"\n"
    "    \"verified purchasers, and I also compared it against several other \"\n"
    "    \"similar products including a smaller 8 ounce travel-sized option from \"\n"
    "    \"a competing brand called NatureNest which costs quite a bit less per \"\n"
    "    \"bottle but noticeably more per ounce once you account for the size, \"\n"
    "    \"so let me know if you would like the full comparison table.\"\n"
    ")\n"
    "\n"
    "for label, text in [('fixture speech', speech_text), ('deliberately too long', too_long_text)]:\n"
    "    secs = tts.estimate_speech_seconds(text)\n"
    "    ok = tts.fits_budget(text)\n"
    "    print(f'{label:22s} ~{secs:5.1f}s  {\"PASS\" if ok else \"FAIL (will be truncated by speak())\"}')"
))

# 9. Synthesize header
cells.append(new_markdown_cell(
    "## 4. Synthesize speech (fragment-based)\n"
    "\n"
    "`speak()` writes a complete audio file and returns its path — the "
    "fragment-based approach from §1. With no `provider` argument and no "
    "`TTS_PROVIDER` env override, this uses the offline `pyttsx3` default "
    "from `cfg.tts_provider`."
))

# 10. Synthesize code
cells.append(new_code_cell(
    "out_path = tts.speak(speech_text)\n"
    "\n"
    "duration_s = tts.estimate_speech_seconds(speech_text)\n"
    "size_kb = out_path.stat().st_size / 1024\n"
    "print('wrote     :', out_path)\n"
    "print(f'~duration : {duration_s:.1f}s (estimated)')\n"
    "print(f'file size : {size_kb:.1f} KB')"
))

# 11. Play inline
cells.append(new_code_cell(
    "Audio(filename=str(out_path))"
))

# 12. Alignment markdown
cells.append(new_markdown_cell(
    "## 5. Align spoken audio with on-screen citations\n"
    "\n"
    "`tts.speak()` only ever voices the Answerer's `speech` field — it never "
    "reads `citations[*].url`/`doc_id` or the full `comparison_table` aloud. "
    "This mirrors the system prompt's voice-style rule "
    "(`prompts/system_assistant.md`, \"Voice style\"): *\"No markdown, emojis, "
    "URLs, or reading out long ingredient lists aloud — those belong on the "
    "screen, not in the speech.\"* Citations and the comparison table are "
    "rendered on screen, next to (not inside) the audio."
))

# 13. Render citations table
cells.append(new_code_cell(
    "import pandas as pd\n"
    "\n"
    "citations_df = pd.DataFrame(example['output']['citations'])\n"
    "comparison_df = pd.DataFrame(example['output']['comparison_table'])\n"
    "\n"
    "display(Audio(filename=str(out_path)))\n"
    "print('Citations (on screen, not spoken):')\n"
    "display(citations_df)\n"
    "print('Comparison table (on screen, not spoken):')\n"
    "display(comparison_df)"
))

# 14. Degradation markdown
cells.append(new_markdown_cell(
    "## 6. Graceful degradation without an API key\n"
    "\n"
    "If `provider=\"openai\"` or `\"elevenlabs\"` is requested but the matching "
    "API key (or, for ElevenLabs, the package) isn't available, `speak()` "
    "logs a warning to stderr and falls back to `pyttsx3` instead of raising "
    "— the same graceful-degradation pattern as Clark's `web_search.py`."
))

# 15. Degradation demo code
cells.append(new_code_cell(
    "os.environ.pop('OPENAI_API_KEY', None)  # ensure the key really is absent for this demo\n"
    "\n"
    "fallback_path = tts.speak(speech_text, provider='openai')\n"
    "print('wrote (via fallback):', fallback_path)\n"
    "Audio(filename=str(fallback_path))"
))

# 16. Batch demo markdown
cells.append(new_markdown_cell(
    "## 7. Batch demo across multiple queries\n"
    "\n"
    "Every example in `answerer_examples.json` — including the ungrounded / "
    "empty-result case — should still produce a playable, on-budget clip."
))

# 17. Batch demo code
cells.append(new_code_cell(
    "manifest = []\n"
    "for ex in examples:\n"
    "    text = ex['output']['speech']\n"
    "    path = tts.speak(text)\n"
    "    manifest.append({\n"
    "        'user_priority': ex['user_priority'],\n"
    "        'n_citations': len(ex['output']['citations']),\n"
    "        'est_seconds': round(tts.estimate_speech_seconds(text), 1),\n"
    "        'fits_budget': tts.fits_budget(text),\n"
    "        'audio_path': str(path),\n"
    "        'size_kb': round(path.stat().st_size / 1024, 1),\n"
    "    })\n"
    "\n"
    "pd.DataFrame(manifest)"
))

# 18. Handoff notes
cells.append(new_markdown_cell(
    "---\n"
    "### Handoff notes\n"
    "\n"
    "* **Signature:** `speak(text: str, out_path: str | Path | None = None, "
    "provider: str | None = None, voice: str | None = None) -> Path`.\n"
    "* **Output location:** defaults to "
    "`rag-system/audio/summary_<content-hash>.wav` (deterministic, no "
    "timestamp — repeat calls with the same text overwrite in place). "
    "`audio/query*.wav` (the ASR notebook's samples) are untouched and stay "
    "git-tracked; `audio/summary_*` is gitignored.\n"
    "* **Optional orchestration:** `04_orchestration.ipynb` can call "
    "`rag.tts.speak(answer['speech'])` on the Answerer's accepted payload to "
    "attach spoken audio to a full agent turn — it is optional because the "
    "text/citations UI already satisfies the pipeline without audio.\n"
    "* **Env vars:** `TTS_PROVIDER` (`pyttsx3` default | `openai` | "
    "`elevenlabs`), `TTS_VOICE`, `TTS_MODEL` (OpenAI TTS model id), plus "
    "`OPENAI_API_KEY` / `ELEVENLABS_API_KEY` only if you opt into a paid "
    "provider. Nothing is required to run this notebook as-is."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT)
