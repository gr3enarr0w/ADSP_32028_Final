# AI Tool Prompt — RAG Assignment Notebook Build

This is the exact prompt used to drive the autonomous build of the assignment notebook. Preserved verbatim here for the assignment's "AI tool prompts used" disclosure requirement, and reused as the actual subagent instruction.

---

<task>
Build ONE Jupyter notebook (Python 3.11) that satisfies a RAG assignment in full: load 3 arXiv papers, chunk, embed, store in ChromaDB, retrieve, generate grounded answers with an open-source LLM, evaluate with RAGAS, manually label groundedness, visualize embeddings via PCA/UMAP, and export to HTML.
</task>

<source_materials>
1. AI Agents Need Memory Control Over More Context — https://arxiv.org/abs/2601.11653
2. ProofAgent Harness: Open Infrastructure for Adversarial Evaluation of AI Agents — https://arxiv.org/abs/2605.24134
3. Agentic Systems: A Guide to Transforming Industries with Vertical AI Agents — https://arxiv.org/abs/2501.00881
</source_materials>

<required_stack>
LangChain, ChromaDB, HuggingFace embeddings, an open-source LLM (no fine-tuning), RAGAS (faithfulness, answer_relevancy, context_precision, context_recall — minimum four metrics), PCA or UMAP for embedding visualization.
</required_stack>

<reuse_from_sibling_project>
A sibling project at /Users/ceverson/Development/hermes-rag-qdrant-efficient-skill already solved several of the hardest parts of this exact problem space. Read its code directly and adapt these patterns — do not reinvent them from scratch, and do not silently ignore the specific bugs/gotchas already found and fixed there:

1. **RAGAS + LangChain version pinning is a real, confirmed failure mode, not hypothetical.** `ragas==0.3.9` requires `langchain-core` on the 0.3.x line (`<0.4.0`). `langchain-ollama==0.3.10` is the confirmed-compatible pin for that line (a plain `pip install langchain-ollama` resolves to 1.1.0+, which breaks everything). Pin versions together and test `import ragas` + your chosen LangChain provider import in the SAME process before building anything on top — see this project's `requirements.txt` comments for the full documented history of this exact conflict.
2. **RAGAS judge LLM needs an explicit, generous `max_tokens`** (this project uses 4096). Leaving it at a wrapper default (often 1024) causes `LLMDidNotFinishException` and silent `NaN` scores scattered across metrics that look unrelated to token limits at first glance.
3. **Three open-source LLMs are already proven live via local Ollama**: `glm-5.2:cloud`, `kimi-k2.7-code:cloud`, `minimax-m3:cloud` (reachable at `http://localhost:11434` via `langchain_ollama.ChatOllama`). Wire ONE of these as the actual GENERATION model for this assignment (not just a judge) — pick whichever produces the most coherent answers in a quick spot check, and say which you picked and why.
4. **Token-based chunking**: use the embedding model's own HuggingFace tokenizer to count tokens for the 200-400 token chunk-size requirement, not a character-count approximation (`len(text)` on characters is NOT tokens). This project's `scripts/utils.py` (`EfficientRAG._token_len`) shows the pattern: `len(tokenizer.encode(text, add_special_tokens=False))`. LangChain's `RecursiveCharacterTextSplitter.from_huggingface_tokenizer(...)` is the direct equivalent for this stack.
5. **PCA/UMAP visualization**: adapt `scripts/visualize_embeddings.py` from the sibling project (built for Qdrant) to pull vectors from ChromaDB instead. Keep its two good habits: report PCA's explained-variance ratio for the first 2 components (don't just show a scatter plot with no quantitative grounding), and scale UMAP's `n_neighbors` down (e.g. `min(15, n_points - 1)`) since its defaults are tuned for large corpora and mislead on small ones (~60-200 chunks expected here across 3 papers).
6. **Manual grounding-label rubric — reuse this exact rubric, do not invent a different one**:
   - **Grounded**: every factual claim in the answer is traceable to the retrieved context.
   - **Partially grounded**: some claims are supported by context, others are unsupported/inferred/from general knowledge.
   - **Not grounded**: the answer contradicts the retrieved context, or ignores it entirely.
7. **Generation prompt policy**: do NOT use a strict "answer ONLY from context, refuse otherwise" prompt — this project measured a 75% refusal rate with that policy on thin/tangential context, which tanked `answer_relevancy` even when retrieval itself was working fine. Instead use attribution-tagged blending: answer fully, tag context-derived claims `[C]` and general-knowledge claims `[G]`, and never fabricate specifics under a `[C]` tag that aren't actually in the context. A "Partially grounded" human label should naturally correlate with answers containing both tags.
8. **`context_recall`/`context_precision` need a `reference` field per RAGAS sample.** This project didn't always have one available; THIS assignment must have one — author a short (1-3 sentence) hand-written reference answer for each of the 12 required questions, grounded in the actual paper content, BEFORE running RAGAS.
</reuse_from_sibling_project>

<genuinely_new_work>
Not covered by the sibling project — build these fresh:
- ChromaDB setup (local, in-notebook, no server) instead of Qdrant.
- LangChain document loaders for the 3 arXiv papers — fetch actual PDF full text (arXiv `/abs/` URLs only show the abstract page; construct or resolve the corresponding `/pdf/` URL and use a PDF loader, e.g. `PyPDFLoader` or equivalent). Verify full-text loading succeeded (print a chunk count and a mid-document sample) before trusting anything downstream — an abstract-only load would silently starve the whole pipeline.
- HuggingFace embeddings via `langchain-huggingface`'s `HuggingFaceEmbeddings` (e.g. `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
- The 12 hand-authored reference answers (point 8 above).
- Single-notebook packaging with the required per-question reporting structure (question / retrieved chunks / generated answer / RAGAS scores / human label / comment).
- HTML export (`jupyter nbconvert --to html`).
</genuinely_new_work>

<required_questions>
1. What is an AI agent?
2. What are the core components of an AI agent?
3. What is a vertical AI agent?
4. How are agentic systems different from traditional LLM applications?
5. Why do AI agents need memory control?
6. What problems happen when agents rely only on long context?
7. What is the Agent Cognitive Compressor?
8. What are the main challenges in evaluating AI agents?
9. What is adversarial evaluation?
10. How does ProofAgent Harness evaluate AI agents?
11. Why is multi-juror scoring useful for agent evaluation?
12. How can RAG help explain agentic AI concepts using source documents?

Note: question 12 is a meta/reflective question, not a fact retrievable from the 3 papers. Expect weak/tangential retrieval for it by design — label it honestly (likely "Partially grounded" or "Not grounded") rather than forcing a grounded-looking answer, and say so in the comment column.
</required_questions>

<constraints>
- Do not fine-tune any model.
- Single notebook, Python 3.11, in its own subfolder — do not touch the sibling project's files, only read from them for reference/reuse.
- Every code cell must actually run top-to-bottom without manual intervention (verify by executing the full notebook, not just individual cells).
- Include an "AI Tool Prompts Used" markdown cell containing this exact prompt, verbatim, per the assignment's disclosure requirement.
</constraints>

<output_format>
One `.ipynb` file plus one `.html` export, both in the new subfolder, structured in the order: setup/imports → load papers → chunk → embed → store in ChromaDB → retrieval function → generation function → per-question loop (12 questions, full reporting table) → RAGAS evaluation → human review labels → embedding visualization + written explanation → AI tool prompt disclosure.
</output_format>
