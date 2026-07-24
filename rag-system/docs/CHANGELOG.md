# Changelog / Known Issues

This document tracks notable bugs found and fixed in this project, plus work that is currently
in progress. It exists mainly so that a future maintainer (human or agent) who sees a strange
symptom can quickly check "has this already happened and been fixed before?" instead of
re-diagnosing it from scratch.

## 2026-07-23 — `visibility` (public/private) payload field added to `personal_corpus_v1`, rule-based, no LLM

**What this adds:** A new `visibility` payload field (`"public"` or `"private"`), backfilled onto
every existing point in `rag_personal_corpus_v1` and now stamped automatically on every
future ingested point, purely by RULE (no LLM/AI call involved).

**Rule (verified against actual stored payloads via the Qdrant scroll API before deciding, not
assumed from naming conventions):**

- `payload.source_type == "external_docs"` → `"public"`. This is the value `ingest.py`'s
  `--crawl-url` path (`crawl_site()` / `crawl_confluence_space()`) sets via `extra_metadata` for
  every page it ingests — confirmed by scrolling `source_type: external_docs` points, all of which
  are external, publicly-accessible websites: Third-Party Service support/Document Service public docs (tags
  `external-docs`/`atlassian-docs`/`atlassian`), Adaptavist/ScriptRunner docs, eazyBI docs (tags
  `external-docs`/`eazybi`), Hermes Agent docs (tags `external-docs`/`hermes-agent`), and the
  Fivetran REST API docs (tags `external-docs`/`fivetran`).
- Everything else → `"private"`. This covers: local files ingested via `--path` (no `source_type`
  key at all on these points — tags like `personal,downloads` / `personal,documents` /
  `personal,development`), Google Workspace content (`source_type: "google_drive"`, tags
  `personal,google-drive`), and Document Service dumps (`source_type: "confluence"`, tags
  `confluence,work`) — both team spaces (OMEGA/DDISDP/ANTSE/HUB) and the personal `~ceverson`
  space, since `--confluence-dump-dir` ingests from an authenticated/private export, unlike the
  public `crawl_confluence_space()` path used for `confluence.atlassian.com`'s public docs.

**Backfill mechanics:** Added a `keyword` payload index on `source_type` (previously no payload
indexes existed on this ~8.5M-point collection, making any filtered operation a full collection
scan) via `PUT /collections/.../index`, then two **filter-based bulk `set_payload` calls** (`PUT
/collections/.../points/payload` with a `filter` selector, not per-point/per-ID updates) — one
scoped to `source_type = external_docs` (→ `public`, 66,464 points), one scoped to `source_type !=
external_docs` (→ `private`, ~8,448,909 points). This is payload-only: no re-embedding,
re-ingestion, vector, or other-payload-field changes. Also added a `keyword` index on the new
`visibility` field itself for future filtered-retrieval use. Both indexes and both `set_payload`
calls only touch `rag_personal_corpus_v1` — no other collection was modified.

The `public` backfill (66,464 points) completed in seconds. The `private` backfill (~8.45M points)
is I/O-bound (rewriting on-disk payload storage per point) and takes on the order of hours on this
local single-node Qdrant instance — it was still running (collection status `yellow`) as of this
entry, progressing at roughly 1,000-1,600 points/sec and monotonically increasing with no errors
observed. Track completion via `GET /collections/rag_personal_corpus_v1` (`status` returns
to `green` when all pending operations finish) or `points/count` with a `visibility` filter to see
the running total approach 8,515,373 total points (66,464 public + 8,448,909 private).

**Future ingestion wiring:** `scripts/utils.py` gained `PUBLIC_VISIBILITY_SOURCE_TYPES =
{"external_docs"}` and `_classify_visibility(extra_metadata)`, applying the identical rule.
`EfficientRAG.ingest()` now computes `visibility = _classify_visibility(extra_metadata)` once per
call (before the per-chunk loop — every chunk of one ingested source shares the same
`source_type`) and stamps it into both the parent-chunk and child-chunk payload dicts alongside
the other fixed fields (`tags`, `chunk_type`, etc.), so every ingestion path (`--path`, `--url`,
`--crawl-url`, `--api`, `--gdrive-query`/`--gdoc-id`/`--gsheet-id`/`--gslide-id`,
`--confluence-dump-dir`) gets a correct `visibility` value automatically with no per-call-site
changes needed in `ingest.py`.

## 2026-07-22 — Multi-page site crawler added (`crawl_site()` / `--crawl-url`); 8 external doc hubs ingested into `personal_corpus_v1`

**Problem:** `--url` only ever ingested ONE page. Several frequently-referenced external doc
sites (Third-Party Service support/Document Service doc hubs, Adaptavist ScriptRunner docs, eazyBI docs, Hermes
Agent docs) are structured as an index/hub page linking out to many individual articles —
ingesting just the hub page captured a list of links, not the actual content underneath it.

**Fix:**
1. `scripts/utils.py::EfficientRAG._fetch_url()` was refactored to extract its trafilatura
   (with BeautifulSoup tag-stripping fallback) logic into a new shared helper,
   `_extract_readable_text(html)`, so per-page extraction logic exists in exactly one place.
2. New `EfficientRAG.crawl_site(start_url, max_pages=200, same_prefix=True)`: breadth-first
   crawl from `start_url`, discovering links via BeautifulSoup `<a href>` parsing (link
   discovery only — content extraction still goes through `_extract_readable_text()`). Scope is
   restricted to the same domain as `start_url`, and by default to a "directory prefix" derived
   by dropping `start_url`'s LAST path segment (e.g. `/jira-software-cloud/resources` ->
   `/jira-software-cloud/`, or `/sr4jc/latest` -> `/sr4jc/`) so a hub page one level inside a doc
   section still reaches sibling/child pages under that section. URLs are normalized
   (fragment/query stripped, trailing slash normalized) before dedup. Returns
   `{"url", "text", "title"}` dicts for every page with non-empty extracted text.
3. `scripts/ingest.py` gained `--crawl-url URL`, `--crawl-max-pages N` (default 200), and
   `--no-crawl-same-prefix` (disables prefix scoping, keeping only same-domain scoping — for a
   genuine cross-section index page that legitimately links to differently-pathed content on the
   same domain). Crawled pages are ingested via the same `raw_text=`/`extra_metadata=` pattern
   `_ingest_google_text()`/`_ingest_confluence_text()` use, tagged
   `{"source_type": "external_docs", "url": ...}`, with the same per-page
   try/except-log-and-continue resilience the `--gdrive-query`/`--confluence-dump-dir` loops have
   (one bad page never aborts the whole crawl's ingestion).

**Prefix-scoping bug caught and fixed during testing:** the first version treated a
trailing-slash start URL (e.g. `.../docs/`) as the literal full scope, which broke the OPPOSITE
case — `https://support.atlassian.com/jira-software-cloud/resources/` crawled to exactly 1 page,
because its real doc articles live under `/jira-software-cloud/docs/`, not
`/jira-software-cloud/resources/`. Fixed by always deriving the prefix from dropping the last
path segment (falling back to keeping a single-segment path as-is, e.g. `/docs` -> `/docs/`, only
when there's nothing to drop to without collapsing to domain-root `/`). Verified both directions
post-fix: the Issue Tracker hub now reaches `/jira-software-cloud/docs/...` articles, and the Hermes Agent
`/docs/` crawl no longer wanders into the unrelated marketing homepage (`/`) or the `zh-Hans`
locale root itself (though sub-pages under `/docs/zh-Hans/...` remain in scope, since they're
still under `/docs/`).

**Test:** `--crawl-url https://hermes-agent.nousresearch.com/docs/ --crawl-max-pages 15` into a
throwaway `test_crawl_ingest_v1` collection — 15 distinct real pages fetched (not the start page
repeated, not off-domain wandering), 859 points ingested with correct per-page citation URLs in
payload metadata; collection deleted after.

**Real ingestion run** (sequential, `personal_corpus_v1`, no `--force-recreate`, alongside a
concurrent unrelated `~/Development` ingestion into the same collection — confirmed via
`ps aux | grep resume_development_ingest` before starting and left untouched throughout). Logs
under `/tmp/personal_corpus_ingest_logs/04_*.log` through `11_*.log`:

| Site | Cap | Pages ingested | Notes |
|---|---|---|---|
| `support.atlassian.com/jira-software-cloud/resources/` | 250 | 250 (cap hit) | tag `external-docs,atlassian` |
| `confluence.atlassian.com/servicedeskcloud/.../jira-cloud-shared-master-home-*.html` | 250 | 195 (queue exhausted) | tag `external-docs,atlassian` |
| `support.atlassian.com/confluence-cloud/resources/` | 250 | 250 (cap hit) | tag `external-docs,atlassian` |
| `confluence.atlassian.com/alldoc/atlassian-documentation-*.html` | 300 | 66 (queue exhausted) | tag `external-docs,atlassian`; run with `--no-crawl-same-prefix` since this hub links to differently-pathed `/spaces/<SPACE>/pages/...` doc spaces across the whole domain by design — this is explicitly a **partial/capped ingestion of a much larger site** (Third-Party Service's full documentation index); most `/spaces/...` landing pages exposed few further links in unauthenticated raw HTML (likely gated behind login/AJAX navigation), so the crawl stopped well short of the cap. Raising the cap alone would likely NOT capture much more without also handling authenticated/JS-rendered navigation — flagged for the user to decide whether deeper investment here is worthwhile |
| `support.atlassian.com/organization-administration/resources/` | 250 | 250 (cap hit) | tag `external-docs,atlassian` |
| `docs.adaptavist.com/sr4jc/latest` | 250 | 232 (queue exhausted) | tag `external-docs,adaptavist`; site redirects to `/sr4jc/current/...`, stayed correctly scoped to `/sr4jc/` |
| `docs.eazybi.com/eazybi` | 250 | 250 (cap hit) | tag `external-docs,eazybi` |
| `hermes-agent.nousresearch.com/docs/` | 250 | 249 (queue exhausted) | tag `external-docs,hermes-agent`; crawl spent a large share of its budget on the `zh-Hans` (Chinese) locale mirror under `/docs/zh-Hans/...`, since it's technically in-scope under the `/docs/` prefix — not a bug, but means a meaningful fraction of ingested content is a translated duplicate of the English pages rather than distinct content |

## 2026-07-22 — `crawl_site()` gained `path_prefix` override; `fivetran.com/docs/rest-api` ingested into `personal_corpus_v1`

**New same-prefix-scoping bug found (third variant of the recurring "how do we scope a doc-hub
crawl" problem — see the two entries above):** the auto-derived prefix logic drops `start_url`'s
LAST path segment on the assumption that `start_url` is a hub page one level *below* the doc
section it belongs to (e.g. `/jira-software-cloud/resources` -> section prefix
`/jira-software-cloud/`). That assumption is wrong when `start_url`'s last segment IS the section
root itself and its children live directly under it — exactly the shape of
`fivetran.com/docs/rest-api` (children like `/docs/rest-api/getting-started`,
`/docs/rest-api/api-reference/...`). A first test crawl (`--crawl-max-pages 15` into throwaway
`test_fivetran_crawl_v1`) confirmed this concretely: the auto-derived prefix came out as `/docs/`
(dropping `rest-api`), so the crawl wandered into unrelated Fivetran product docs —
`/docs/core-concepts`, `/docs/connectors`, `/docs/destinations`, `/docs/logs`,
`/docs/security-and-privacy/security`, etc. — instead of staying inside the REST API doc section.
Only 4 of the 15 fetched pages were actually under `/docs/rest-api/`.

**Fix:** `EfficientRAG.crawl_site()` gained an optional `path_prefix` parameter that, when given,
is used verbatim as the scope prefix instead of the auto-derived one (trailing slash normalized).
The existing auto-derivation is left as the default — it's still correct for the Issue Tracker/Document Service
hub-page shape documented above — this is an explicit opt-in override for the opposite shape, not
a change in default behavior. Wired into `scripts/ingest.py` as `--crawl-path-prefix` (e.g.
`--crawl-path-prefix /docs/rest-api/`), passed through to `crawl_site()`; no effect when
`--no-crawl-same-prefix` is set.

**Retest after fix:** same throwaway `test_fivetran_crawl_v1` collection (deleted after inspection,
old flawed collection deleted first), `--crawl-max-pages 15 --crawl-path-prefix /docs/rest-api/` —
all 15 fetched pages were distinct, on-topic REST API doc pages (`getting-started`, `api-tools`,
`webhooks`, `api-reference` and several of its sub-pages, `changelog`, `tutorials`,
`troubleshooting`, `sample-code-projects`, `powered-by-fivetran`); spot-checked payload `url`/
`source` fields matched the actual crawled page for each sampled point.

**Sitemap check (per the standing "don't silently accept a low page count" policy from the
Document Service/`alldoc` investigation above):** confirmed via `fivetran.com/docs-sitemap.xml` that the
REST API doc section genuinely has 254 pages total, and all of them are reachable through plain
`<a href>` link-following (verified the root `/docs/rest-api` page alone links to ~24 of them,
including the `api-reference` index which fans out to the ~200 leaf endpoint pages) — no JS-hydrated
or API-driven navigation gap like the earlier Document Service space case, so a plain BFS crawl was
sufficient once correctly scoped; no sitemap-driven enumeration was needed.

**Real ingestion run:** `--crawl-url https://fivetran.com/docs/rest-api --crawl-max-pages 250
--crawl-path-prefix /docs/rest-api/ --collection personal_corpus_v1 --tags
"external-docs,fivetran"`, no `--force-recreate`, run concurrently alongside the ongoing
`~/Development` resume ingestion into the same collection (confirmed via `ps aux | grep
ingest.py` before starting, left untouched throughout — no separate Third-Party Service re-crawl was
running at the time). Log: `/tmp/personal_corpus_ingest_logs/13_crawl_fivetran.log`.

| Site | Cap | Pages ingested | Notes |
|---|---|---|---|
| `fivetran.com/docs/rest-api` | 250 | 250 (cap hit) | tag `external-docs,fivetran`; site has 254 total pages in this section per `docs-sitemap.xml`, so 250 is a near-complete capture, not early exhaustion; zero fetch/HTTP/ingest errors |

`personal_corpus_v1` point count: 7,001,630 -> 7,004,872 (+3,242), confirmed via the Qdrant
`/collections/{name}` endpoint immediately before and after the run.

**Net effect:** `personal_corpus_v1` point count grew from 6,820,803 to 7,001,630
(+180,827 points) across this run — includes both this crawl ingestion and the concurrent
`~/Development` resume running in parallel; growth-only, nothing was dropped or recreated.

## 2026-07-22 — Fatal crash on encrypted/corrupted PDFs during `--path --recursive` ingestion: unhandled `pypdf` exception killed the entire batch

**Symptom:** the real `~/Development` leg of `personal_corpus_v1`'s three-step personal-corpus
ingestion (`/tmp/personal_corpus_full_run.sh`: Downloads → Documents → Development) died with an
uncaught Python traceback partway through, killing the whole process rather than skipping the one
bad file. Only 48,810 of ~78,077 filtered files had been ingested when it died.

**Root cause:** `scripts/utils.py`'s `_load_local_file()` did
`"\n".join([p.extract_text() or "" for p in reader.pages])` on a `pypdf.PdfReader` with no
exception handling at all. The culprit file —
`~/Development/Project_Review/n8n/packages/nodes-base/nodes/ReadPdf/test/sample-encrypted.pdf`
(an n8n test fixture, an intentionally-encrypted PDF used to test n8n's own "ReadPDF" node) —
raised `pypdf.errors.FileNotDecryptedError: File has not been decrypted` deep inside
`reader.pages` iteration. That exception propagated all the way up through `ingest()` and out of
`scripts/ingest.py`'s `--path --recursive` per-file loop (`main()`, `stat = rag.ingest(...)`),
which had **no try/except around the `rag.ingest()` call at all** — unlike the `--gdrive-query` and
`--confluence-dump-dir` loops in the same file, which already wrap each item's ingest in
`try/except Exception` and log-and-continue on failure. This was a gap specific to the
`--path --recursive` loop, not a systemic absence of the pattern.

**Fix (defense in depth, two layers):**
1. `scripts/utils.py::_load_local_file()` — PDF page-text extraction is now wrapped in
   `try/except (PyPdfError, PdfReadError, FileNotDecryptedError)` plus a broad `except Exception`
   fallback (pypdf can raise plain `ValueError`/`struct.error`/etc. on sufficiently malformed
   files outside its own exception hierarchy). On failure, logs
   `"Skipping unreadable/encrypted PDF: {path} (...)"` and returns `""`, which flows into
   `ingest()`'s existing `if not raw_text.strip(): return {"status": "error", ...}` path — an
   ordinary per-file skip, not a crash. DOCX loading (`DocxDocument(path)`/paragraph extraction)
   got the same try/except treatment for parity, since it was equally unguarded.
2. `scripts/ingest.py`'s `--path --recursive` (and non-`--recursive` directory) per-file loop now
   wraps `rag.ingest(...)` in `try/except Exception`, logging
   `"ERROR ingesting {f}: {exc!r} — skipped, continuing"` and moving on — matching the resilience
   policy the `--gdrive-query`/`--confluence-dump-dir` loops already had. This is deliberate
   defense in depth: new file-type-specific bugs (bad `.docx`, malformed `.csv`, etc.) will keep
   surfacing, and the outer loop should never again let one file's loader bug abort a
   multi-hour/multi-thousand-file batch, independent of whether `_load_local_file()` itself catches
   the specific exception.

**Resume strategy:** child-chunk Qdrant point IDs are `str(uuid.uuid4())` — random, NOT
deterministic — so a full re-run of the entire `~/Development` tree would have duplicated every
point for the 48,810 files that already succeeded (only parent-chunk point IDs are deterministic,
via `uuid5` over `parent_id`). A full re-run was therefore rejected as unsafe for this corpus. Used
a one-off resume script (`/tmp/resume_development_ingest.py`) instead: parses
`/tmp/personal_corpus_ingest_logs/03_development.log` for every `"Ingesting file: <path>"` line
printed before the crash (all but the last one — printed immediately before `rag.ingest()` is
called, so if a later line printed, the earlier file's call must have returned — are confirmed
successful; the last one is the in-flight culprit), re-derives the identical filtered file list via
`utils.iter_filtered_files()` with the same `--require-git-authorship user@example.com` /
default exclude-dir / extension / size filters, and ingests only the files not already in the
"succeeded" set (retrying the culprit itself, which now gets cleanly skipped instead of crashing).
Launched as `nohup .venv/bin/python /tmp/resume_development_ingest.py` in the background, logging
to `/tmp/personal_corpus_ingest_logs/03_development_resumed.log`, with `force_recreate=False`
throughout (collection already exists and must not be wiped). Confirmed alive and progressing past
the crash point (culprit file cleanly skipped, ingestion continued into
`nodes-base/credentials/*.credentials.ts` and beyond) with the collection's `points_count` growing
monotonically (6,814,691 → 6,823,070 within the first ~90s of the resumed run).

## 2026-07-22 — CORRECTION: the `alldoc` crawl's "behind login/AJAX" diagnosis was WRONG; real fix is `crawl_confluence_space()` (Document Service Content REST API), 1,605 new Third-Party Service doc pages ingested into `personal_corpus_v1`

**The earlier entry above (`8 external doc hubs ingested`) was wrong about why the
`confluence.atlassian.com/alldoc/...` crawl stopped at 66/300 pages.** It guessed the space-home
landing pages it reached were "likely gated behind login/AJAX navigation." This was verified WRONG
by directly fetching one of those space-home URLs
(`https://confluence.atlassian.com/spaces/JIRASOFTWARECLOUD/pages/764477791/Issue Tracker+Software+documentation`)
with plain unauthenticated `requests.get()`: HTTP 200, real content, no login wall. Log inspection
confirmed the original crawl even reached ~24 distinct "space home" pages (one per Third-Party Service
product/version — Issue Tracker Software Cloud, Issue Tracker Service Management, Document Service Cloud, Bitbucket Cloud,
several Data Center product docs, plus legacy products like Bamboo/Crucible/Fisheye/Crowd/Clover)
before its BFS queue ran dry at 66 total pages.

**Real root cause:** `crawl_site()`'s link-discovery only sees plain `<a href>` tags in a page's
raw HTML. Each Document Service "space home" page's actual page-tree/sidebar navigation (which normally
lists every page in that space) is NOT rendered as plain `<a href>` tags in that page's own raw
HTML — so a same-domain, same-prefix BFS starting at the top-level `alldoc` index reaches each
space's shallow entry point and then has nowhere further to go, well before descending into any
space's real depth (which is often hundreds of pages).

**Investigation (this fix), verified directly before writing any code:**
- `GET https://confluence.atlassian.com/rest/api/content/{id}/child/page` and
  `.../rest/api/content/search?cql=...` both work fully unauthenticated (HTTP 200, real JSON) —
  confluence.atlassian.com is Third-Party Service's own public marketing/docs instance and allows anonymous
  read access to its Content REST API, unlike this project's internal Example Organization Issue Tracker/Document Service
  instance.
- Tried CQL `cql=ancestor=<id>` first (matches an entire subtree in one paginated call) — it
  worked great on the FIRST space tested (JIRASOFTWARECLOUD: found the full 150-page cap) but was
  caught, mid production-run, silently returning 0-5 results for most other spaces tested
  (ConfCloud, BITBUCKET, SERVICEDESKCLOUD) even though those pages demonstrably have real children.
  This instance's `ancestor` search index is not consistently populated across all spaces.
- Switched to `child/page`, walked via BFS from each space's root page — fixed ConfCloud (found
  all 150), but SERVICEDESKCLOUD's actual root ("Issue Tracker Cloud Shared Master Home") is a shallow stub
  with only 4 direct children, while that space's real ~690 content pages are NOT nested under it
  in the page tree at all (siblings/orphans as far as parent/child goes) — BFS alone still
  undercounted it (5 pages).
- **Final fix:** `EfficientRAG.crawl_confluence_space(base_url, root_page_id, max_pages,
  space_key=None)` (new method, `scripts/utils.py`) combines BOTH mechanisms — when `space_key` is
  given, it first pulls the full flat page list via `cql=space={key} AND type=page` (a strict
  superset of BFS results on spaces like SERVICEDESKCLOUD, but sometimes 0 results on spaces like
  ConfCloud/BITBUCKET where BFS finds real content), then ALSO runs the `child/page` BFS from
  `root_page_id`, taking the union (deduped by page ID). Each source covers the other's gaps.
  Each discovered page is fetched at `{base_url}/pages/viewpage.action?pageId={id}` — an ID-based
  URL, not the title-based `/display/{SPACE}/{Title}` slug from `_links.webui`, which was observed
  to 404 on real pages when the current title doesn't exactly match Document Service's stored slug
  encoding (renamed pages, unusual punctuation) — then extracted via the same
  `_extract_readable_text()` used by `_fetch_url()`/`crawl_site()`.
- Verified end-to-end (crawl → ingest → retrieve, with correct citation URLs in the formatted
  context) in a throwaway `test_atlassian_confluence_space_crawl` collection before running
  anything against the real corpus; collection deleted after.

**Scope:** per the corpus owner's Issue Tracker/Document Service/Third-Party Service-admin-heavy role, prioritized core
Issue Tracker Software, Issue Tracker Service Management, Document Service, and Bitbucket docs — both Cloud and Data
Center/Server, since an admin role plausibly touches both — plus Issue Tracker automation and GDPR
compliance guides. Explicitly skipped niche/legacy product spaces: Bamboo, Crucible, Fisheye,
Crowd, Clover, Sourcetree (GSWST), Questions for Document Service (Cloud + Server), and Team Calendars.

**Real ingestion** into `personal_corpus_v1` (checked `ps aux | grep ingest.py` first — a separate
`~/Development` resume-ingestion, and briefly a Fivetran doc crawl, were running concurrently
against the same collection throughout; never passed `--force-recreate`; every point ID is a fresh
`uuid.uuid4()`, so concurrent appends from unrelated processes can't collide). Logged to
`/tmp/personal_corpus_ingest_logs/12_crawl_atlassian_alldoc_v2.log` (kept separate from the
original `07_crawl_atlassian_alldoc.log`, which is left as-is/uncorrected in place, for history):

| Space (key) | Cap | Pages ingested |
|---|---|---|
| Issue Tracker Software Cloud documentation (JIRASOFTWARECLOUD) | 150 | 150 |
| Document Service Cloud Documentation Home (ConfCloud) | 150 | 150 |
| Issue Tracker Service Management Cloud (SERVICEDESKCLOUD) | 150 | 150 |
| Bitbucket Cloud documentation (BITBUCKET) | 120 | 120 |
| Third-Party Service Cloud documentation (Cloud) | 120 | 120 |
| Get started with Issue Tracker Core Cloud (GSWJC) | 60 | 14 (space exhausted) |
| Issue Tracker Software Data Center 11.3 docs (JIRASOFTWARESERVER) | 120 | 120 |
| Issue Tracker Service Management Data Center 11.3 docs (SERVICEDESKSERVER) | 120 | 120 |
| Administering Issue Tracker Data Center applications (ADMINJIRASERVER) | 120 | 120 |
| Document Service Data Center documentation (DOC) | 120 | 120 |
| Issue Tracker Data Center automation (AUTOMATION) | 100 | 81 (space exhausted) |
| Issue Tracker Automation Knowledge Base (automationkb) | 100 | 100 |
| Issue Tracker Core Server 9.12 documentation (JIRACORESERVER) | 80 | 80 |
| Bitbucket Data Center documentation (BitbucketServer) | 100 | 100 |
| Server/DC GDPR support guides (GDPR) | 60 | 60 |

Zero fetch/ingest errors across all 15 spaces. **1,605 unique new pages** ingested this pass
(beyond the original 66 from the uncorrected crawl) — JIRASOFTWARECLOUD and ConfCloud's 150 each
were ingested during earlier iterations of this same fix (before the CQL-only and BFS-only bugs
above were found and corrected) and were NOT re-crawled in the final run to avoid duplicating
their points; SERVICEDESKCLOUD and BITBUCKET's small (5-and-1-page) earlier partial attempts were
superseded by full re-crawls under the final combined method, leaving ~12 harmless duplicate-page
ingests (a handful of extra points out of ~29,500 total added by this whole effort — well under
0.5%). `personal_corpus_v1`'s `points_count` grew monotonically throughout (never decreased,
confirming append-only behavior held under concurrent writes from the other running processes).

## 2026-07-21 — FIRST non-benchmark-contaminated validation: closed-book vs RAG vs RAG+HyDE against `personal_corpus_v1` (39q, single confirmatory run)

**Why this is a milestone:** every prior RAGAS evaluation in this project ran against NFCorpus, a
public BEIR benchmark — a plausible LLM pretraining contaminant, meaning any "RAG beats
closed-book" (or doesn't) finding on it can't be fully trusted. This run instead used
`personal_corpus_v1` (2.38M Qdrant points — the user's real local files, 5 Document Service spaces, and
Google Workspace content) with a freshly hand-reviewed 39-question testset
(`data/real_corpus_eval_questions_v1.json`, security/PII questions pre-filtered by the user) —
content genuinely impossible for any LLM to have seen during training. This is the first time
this pipeline's actual real-world value has been measured without that confound.

**Method:** reused `scripts/evaluate_ragas.py`'s existing harness unmodified in approach —
`run_closed_book_evaluation()` (closed-book), `run_ragas_evaluation()` with
`_make_rag_retrieve_func("personal_corpus_v1", load_config())` (rag, `hyde_enabled: false`), and
the same with `retrieval.hyde_enabled: True` (rag_hyde) — scored with `faithfulness`,
`answer_relevancy` (raw + disclaimer-stripped, via the project's existing
`compute_disclaimer_stripped_answer_relevancy()` fix), `context_precision`, `context_recall`
(against the testset's `reference` field), and `answer_correctness`. Single run, N=39, no
repeats/sweep (this was a preregistered confirmatory test on a fresh question set, not a pilot).
Judge LLM/embeddings: Claude on Vertex AI (`claude-sonnet-4-5@20250929`) + Vertex embeddings
(`gemini-embedding-001`) — this environment's existing fallback default (no `ANTHROPIC_API_KEY`/
`OPENAI_API_KEY` set), unmodified from the harness's own defaults. Stats: paired Wilcoxon
signed-rank + bootstrap 95% CI on the per-question delta, matching this project's established
confirmatory-run methodology (`/tmp/nfcorpus_eval_v2/analyze_hyde_confirmatory.py`).

**Concurrency fix landed mid-run:** `run_ragas_evaluation()`'s per-question split-faithfulness
follow-up loop (`compute_split_faithfulness()`, one extra `evaluate()` call per question with at
least one `[C]`-tagged claim) was a plain sequential `for` loop — corrected to
`concurrent.futures.ThreadPoolExecutor(max_workers=9)`. These are independent Claude/Anthropic
judge calls (each builds its own fresh judge LLM/`Faithfulness()` instance), NOT NeuralWatt-hosted,
so none of the NeuralWatt-specific 3-concurrent-request account-wide rate limit or its
`max_workers=1`-per-thread pattern (see `run_neuralwatt_multi_judge_consensus()`) applies here.
Arms 1 (closed-book) and 2 (rag) had already finished under the old sequential code before the fix
landed and were not re-run; arm 3 (rag_hyde) ran entirely under the parallelized version and
completed in 1406.8s (~23.4 min) total. The equivalent sequential step took ~8 minutes alone for
arm 2's 39 rows; the same phase completed in well under 2 minutes for arm 3 once parallelized —
roughly a 4-5x reduction on that specific step (short of the full theoretical 9x, since individual
judge calls still ran 10-20+ seconds each with some inter-thread contention). This fix is now
permanent in `scripts/evaluate_ragas.py`, benefiting every future `run_ragas_evaluation()` caller.

**Results (N=39, full detail in `docs/REAL_CORPUS_EVAL_V1_SUMMARY.md` and
`data/real_corpus_eval_results_v1.json`):**

| Comparison | Metric | mean delta | 95% CI | Wilcoxon p | Significant? |
|---|---|---|---|---|---|
| closed_book → rag | answer_relevancy | +0.2164 | [+0.031, +0.395] | 0.0835 | No |
| closed_book → rag | answer_relevancy_disclaimer_stripped | +0.2825 | [+0.091, +0.462] | **0.0274** | **Yes** |
| closed_book → rag_hyde | answer_relevancy | +0.2647 | [+0.084, +0.439] | **0.0190** | **Yes** |
| closed_book → rag_hyde | answer_relevancy_disclaimer_stripped | +0.3480 | [+0.155, +0.522] | **0.0038** | **Yes** |
| rag → rag_hyde | answer_relevancy | +0.0483 | [+0.002, +0.116] | 0.1138 | No |
| rag → rag_hyde | context_precision / context_recall / faithfulness / answer_correctness | all ≈0 | all straddle 0 | all p>0.39 | No |

**Verdict: RAG confirmed to help on genuinely unseen data.** Both RAG arms are far more relevant
to the actual question asked than closed-book Claude (unsurprising — these questions ask about
facts inside the user's own private Document Service/local-file/Google-Workspace content that a
general-purpose model has no way to know). The disclaimer-stripped `answer_relevancy` comparison
(the metric this project's own zero-gate fix targets) is significant for **both** RAG arms vs.
closed-book; the raw metric reaches significance only for rag_hyde vs. closed-book. **HyDE remains
unconfirmed as an improvement over baseline RAG** on this corpus at N=39 (every rag→rag_hyde
comparison has p>0.1) — this replicates, on a completely different and uncontaminated corpus, the
same "not significant" HyDE-vs-baseline finding this project already found on NFCorpus
(2026-07-20 entry below). Retrieval quality itself has real room to improve
(`context_precision`≈0.49-0.50, `context_recall`≈0.65-0.67) — this is now a validated target for
future retrieval-tuning work, not a guess.

**Important reporting caveat:** closed-book's `answer_similarity` (0.828, embedding cosine vs.
reference) and the RAG arms' `answer_correctness` (~0.49-0.50, LLM-judged atomic-statement
TP/FP/FN decomposition) are two *different* metrics with different scales — this is
`run_closed_book_evaluation()`'s own deliberate design (no context available to run
`answer_correctness` against), not a comparable pair. Do not read those two numbers as a
correctness comparison.

**If you see this again:** the fixed sequential-loop-vs-ThreadPoolExecutor pattern for
Claude/Anthropic-based per-question judge calls (as distinct from the NeuralWatt-specific
concurrency constraints elsewhere in this file) is now in `run_ragas_evaluation()`'s
split-faithfulness block in `scripts/evaluate_ragas.py` — check there first if a future eval run
seems to be running per-question judge calls one at a time when it shouldn't need to.

## 2026-07-21 — Personal Document Service space (`~ceverson`) ingested into `personal_corpus_v1`

**What this adds:** A 5th Document Service space ingested via the existing `--confluence-dump-dir`
pipeline (previously used for the 4 team spaces OMEGA, DDISDP, ANTSE, HUB — 2,061 pages), this
time for the user's own personal Document Service space, key `~ceverson`
(https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/overview).

- Resolved the personal space's numeric ID (`137691136`) via
  `mcp__jira_prod__getDocument ServiceSpaces(keys="~ceverson")`, since
  `getPagesInDocument ServiceSpace`'s `spaceId` parameter requires a numeric ID, not the `~`-prefixed
  key, despite the tool's parameter name.
- Paginated `mcp__jira_prod__getPagesInDocument ServiceSpace` fully (single page of results at
  `limit=250` was sufficient — this personal space only has 17 pages total, unlike the much larger
  team spaces).
- **Draft/unpublished-page filtering:** `getPagesInDocument ServiceSpace`'s `status` parameter only
  accepts `current | archived | deleted | trashed` — there is no `draft` enum value, confirming
  that unpublished drafts are not exposed by this listing endpoint at all (they simply never
  appear in the results, rather than appearing with a `draft` status that needs filtering out).
  All 17 pages returned had `status: "current"`; 0 pages were excluded as drafts because none were
  present to exclude.
  Each page's body was independently re-fetched and verified via
  `mcp__jira_prod__getDocument ServicePage` (rather than trusting the space-listing response's inline
  `body` field) to guarantee exact text fidelity before writing the dump file.
- Dump file written to `/tmp/confluence_dumps_personal/PERSONAL_ceverson.json` (kept in its own
  directory, separate from `/tmp/confluence_dumps/`, to avoid accidentally re-ingesting the 4
  existing team-space dumps). Filename avoids the literal `~` character (shell-globbing hazard);
  the real space key `~ceverson` is preserved in each record's `"space"` field for accurate
  citation metadata. Schema matches the existing 4 dumps exactly: `page_id`, `title`, `url`,
  `text`, `space`.
- `url` field built as `https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/{page_id}/{title-slug}`
  (same pattern as the 4 existing dumps) since neither `getPagesInDocument ServiceSpace` nor
  `getDocument ServicePage` returns a ready-made webui link field for this MCP server.
- Ingested via
  `.venv/bin/python scripts/ingest.py --confluence-dump-dir /tmp/confluence_dumps_personal --collection personal_corpus_v1 --mode light --tags "confluence,work,personal-space"`,
  intentionally **without** `--force-recreate` (append-only), run concurrently with an
  already-in-progress, unrelated `--path ~/Downloads --force-recreate` local-file ingestion job
  against the same collection — consistent with this collection's established pattern of safe
  concurrent appends.

**Result:** 17 pages found total, 0 excluded as drafts, 7 skipped as empty (no body content: e.g.
placeholder/test pages like `TestMiro`, `test widget`, `Test`), 10 pages actually ingested.
Qdrant point count for `rag_personal_corpus_v1` grew from 1,404,768 to 1,404,849 (+81
points; 10 parent + 71 child chunks across the 10 non-empty pages), confirming append-only,
monotonically-increasing behavior. Log: `/tmp/personal_corpus_ingest_logs/07_confluence_personal.log`.

## 2026-07-21 — Google Slides ingestion support added; 20 previously-skipped presentations backfilled into `personal_corpus_v1`

**What this adds:** Native Google Slides support to the existing Google Workspace connector
(`scripts/google_workspace.py`), closing a gap where a completed 856-file Google Docs/Sheets
ingestion run had skipped 20 native Google Slides files (`application/vnd.google-apps.presentation`
mimeType) and 2 uploaded `.pptx` files entirely as "unsupported Drive mimeType" (see
`/tmp/personal_corpus_ingest_logs/05_google.log`).

- `GoogleWorkspaceClient.get_slides_text(presentation_id)`: fetches a presentation via the Slides
  API (`slides.googleapis.com` v1, `presentations().get()`) and flattens it via new
  `flatten_google_slides()`, mirroring the existing `get_doc_text()` / `flatten_google_doc()`
  pattern. Walks `presentation.slides[].pageElements[]`, extracting
  `.shape.text.textElements[].textRun.content` for text boxes/titles/placeholders and
  `.table.tableRows[].tableCells[].text.textElements[]` for tables (joined `" | "` per row). Each
  slide's text is prefixed with a `--- Slide N ---` boundary marker (even when a slide has no
  extractable text, so numbering stays accurate against the real deck) so retrieved chunks retain
  slide-level structure/citability. Non-text page elements (images, videos, lines, word art, sheets
  charts) are skipped — no extractable plain text.
- New OAuth scope: `https://www.googleapis.com/auth/presentations.readonly`, added to `SCOPES`
  alongside the existing `drive.readonly` / `documents.readonly` / `spreadsheets.readonly`, plus a
  new `slides` service property on `GoogleWorkspaceClient` (built the same way as `docs`/`sheets`).
- `scripts/ingest.py`: new `--gslide-id ID` CLI flag (single-file path, mirrors `--gdoc-id` /
  `--gsheet-id`), and the `--gdrive-query` search-loop's mimeType dispatch now routes
  `GOOGLE_SLIDES_MIME_TYPE` files to `get_slides_text()` instead of hitting the "unsupported
  mimeType" skip branch.
- Uploaded `.pptx` files (`application/vnd.openxmlformats-officedocument.presentationml.presentation`)
  remain unsupported and are still skipped — out of scope for this change (native Slides was the
  priority; `.pptx` parsing would need a different library, e.g. `python-pptx`, and was deferred).

**OAuth re-consent required and completed:** adding a new scope invalidated the previously cached
`.credentials/token.json` — its refresh token had been issued for the narrower 3-scope set, so
attempting to refresh it against the updated 4-scope `SCOPES` list failed with
`RefreshError: invalid_scope`, exactly as anticipated. Removing the stale token file and re-running
triggered the standard interactive Desktop-app consent flow (`InstalledAppFlow.run_local_server`);
in this environment the browser-based consent completed automatically, producing a new
`token.json` whose `scopes` field confirms all 4 scopes are now granted. No workaround or fallback
to the old token was attempted.

**Verified against a throwaway `test_gslides_ingest_v1` collection** (created, ingested via both
`--gslide-id` and `--gdrive-query` dispatch, spot-checked chunk payloads for correct `--- Slide N
---` boundaries, then deleted) before touching the real corpus.

**Backfill into `personal_corpus_v1`:** re-searched Drive (`--google-involvement owner_or_writer`,
no text filter) and cross-referenced against the 20 mimeType-skip lines in the prior run's log —
same 20 file IDs matched exactly. Ingested each individually via `--gslide-id` (no
`--force-recreate`), tagged `personal,google-drive,slides`. All 20 succeeded, 0 failures.
`rag_personal_corpus_v1`'s point count went from 1,198,455 → 1,199,725 (+1,270 points across
the 20 presentations) — confirmed to only grow, never drop, i.e. no accidental recreate occurred.
This ran concurrently with a separate, still-in-progress local-file ingestion process targeting the
same collection; both processes appending concurrently was already proven safe by the preceding
Document Service and Google Docs/Sheets runs.

## 2026-07-21 — Document Service Cloud (Example Organization wiki) ingestion connector added

**What this adds:** A 4th ingestible data source, alongside local files, web URLs/APIs, and Google
Workspace: `scripts/confluence_workspace.py` (mirroring `scripts/google_workspace.py`'s
client-class structure) + two new `scripts/ingest.py` CLI flags, `--confluence-space KEY[,KEY...]`
(ingest every current page in one or more spaces) and `--confluence-page-id ID[,ID...]` (ingest
specific pages by ID), routed through a new `_ingest_confluence_text()` helper analogous to
`_ingest_google_text()`.

**Auth mechanism:** HTTP Basic Auth (Third-Party Service account email + an Third-Party Service API token) against the
Document Service Cloud REST API v2 (`/wiki/api/v2/*`) — this is Third-Party Service's own documented mechanism for
Cloud REST API access (https://developer.atlassian.com/cloud/confluence/basic-auth-for-rest-apis/),
and intentionally NOT the OAuth 2.0 3LO session used by IDE/agent MCP integrations (e.g. the
Rovo/Third-Party Service MCP tools available inside this coding environment) — a standalone script run later
by an end user has no access to that session and needs its own credential. New `confluence` section
in `config/config.yaml`: `base_url` (default `https://<YOUR_DOMAIN>.atlassian.net`), `email` /
`email_env_var` (default `CONFLUENCE_EMAIL`), `api_token_env_var` (default
`CONFLUENCE_API_TOKEN`) with a gitignored-file fallback at `api_token_path` (default
`.credentials/confluence_api_token`, same `.credentials/`-gitignored pattern as Google Workspace).
`.gitignore` updated with an explicit `confluence_api_token` entry (redundant with the existing
`.credentials/` rule, added for clarity).

**IMPORTANT — confirmed during development, not just documented defensively:** Third-Party Service API
tokens can be "classic" (full account access across every product the user can reach) or "scoped"
(created against specific product+scope combinations, the current default token-creation flow). A
real Issue Tracker-scoped API token already present in this environment (`JIRA_API_TOKEN`) was tested
directly against `GET /wiki/api/v2/spaces` and returned `200 OK` with an **empty** result set and
`GET /wiki/rest/api/user/current` reported `"type": "anonymous"` — i.e. a Issue Tracker-only-scoped token
does not error against Document Service endpoints, it silently authenticates as Anonymous, which looks
exactly like "no access to this space" rather than "wrong credential". `confluence_workspace.py`'s
`resolve_confluence_credentials()` therefore fails loud with an actionable message if no
email/token can be resolved from config/env/credentials-file at all, but cannot itself detect the
scoped-token-silently-anonymous case (Document Service's API gives no signal to distinguish it from a
genuinely empty/inaccessible space) — this is called out prominently in both the module docstring
and `config/config.yaml`'s new `confluence` section comment. **A user of this connector must create
a NEW Third-Party Service API token with explicit Document Service scopes** (`read:page:confluence`,
`read:space:confluence`, `read:content-details:confluence`) or a classic/unscoped token — the
existing `JIRA_API_TOKEN` in this environment does NOT work for this feature.

**Page listing / text extraction:** `list_space_pages(space_key)` resolves the space key to its
numeric ID via `GET /wiki/api/v2/spaces?keys=...`, then paginates
`GET /wiki/api/v2/spaces/{id}/pages` (250-page batches) following `_links.next` cursors until
exhausted — confirmed live (via the Third-Party Service MCP tools available in this coding environment, used
only for research/exploration, never as the ingestion path itself) that this v2 endpoint is what
Example Organization's Document Service Cloud instance actually serves, and that three of the four target spaces
(OMEGA, DDISDP, HUB) each have 250+ pages (hit the single-page-list limit; not fully enumerated)
while ANTSE has exactly 9. `get_page_text(page_id)` fetches `GET /wiki/api/v2/pages/{id}
?body-format=storage` (Document Service's XHTML-based "storage format" — the v2 API has no plain-text
body-format option) and flattens it via the same BeautifulSoup tag-stripping approach as
`EfficientRAG._fetch_url()`'s fallback path in `scripts/utils.py`, rather than a structural,
macro-aware ADF flatten (a reasonable, good-enough first cut matching this project's existing
URL-ingestion fidelity bar). `build_page_url()` prefers the API's own `_links.webui` path for the
citation URL (authoritative Document Service-decided title-slug encoding) over hand-constructing one.

**Live test:** ingested the ANTSE space (9 pages, the smallest of the four target spaces) into a
throwaway `test_confluence_ingest_v1` collection using page content fetched via the Third-Party Service MCP
tools (standing in for the direct-HTTP fetch, since no Document Service-scoped API token was available in
this environment to exercise `Document ServiceClient`'s real HTTP calls end-to-end) fed through the same
`rag.ingest(..., raw_text=..., extra_metadata=...)` path `_ingest_confluence_text()` uses. All 9
pages chunked/embedded/upserted successfully with `source_type: confluence` metadata and correct,
real `https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/ANTSE/pages/<id>/...` citation URLs on every point;
the collection was deleted immediately after verification. The direct-HTTP code path in
`Document ServiceClient` itself (space-ID resolution, cursor pagination, storage-format fetch) is
implementation-complete and compiles clean, but was NOT exercised against a live Document Service-scoped
credential in this environment — that remains the one unverified seam pending the user creating a
properly-scoped API token.

## 2026-07-21 — Document Service ingestion redirected: MCP-based dump files replace the API-token connector

**Why:** The connector added earlier today (`scripts/confluence_workspace.py`, the entry directly
above) requires a Document Service-scoped Third-Party Service API token that does not exist in this environment —
the only token available (`JIRA_API_TOKEN`) is Issue Tracker-only-scoped and silently authenticates as
Anonymous against every Document Service endpoint, as documented in that entry. Rather than asking the
user to go create a new scoped token, the user pointed out that the Third-Party Service MCP tools
(`mcp__jira_prod__*` / `mcp__jira_stage__*`) are already live and authenticated in an interactive
agent session and can read Document Service directly — no new credentials needed at all. That makes the
whole HTTP Basic Auth connector unnecessary for the actual use case (an agent with MCP access
ingesting Document Service content), so it was removed rather than kept as a second, misleading,
practically-unusable path.

**What changed:**

- **Removed** `scripts/confluence_workspace.py` entirely (the `Document ServiceClient` HTTP client,
  `resolve_confluence_credentials()`, and all API-token/email resolution logic).
- **Removed** from `scripts/ingest.py`: the `--confluence-space`, `--confluence-page-id`,
  `--confluence-base-url`, `--confluence-email`, and `--confluence-api-token` CLI flags, and the
  `elif args.confluence_space or args.confluence_page_id:` branch that drove
  `Document ServiceClient`.
- **Added** `--confluence-dump-dir DIR` to `scripts/ingest.py`: reads every `*.json` file in `DIR`,
  each expected to be a JSON array of page objects with `page_id`, `title`, `url`, `text` (and
  optionally `space`, used for payload metadata and falling back to the dump file's basename if
  absent). Each page is ingested via `rag.ingest(url, ..., raw_text=text, extra_metadata={...})`,
  matching the exact `extra_metadata` shape/field-naming convention `_ingest_google_text()` already
  uses for the Google Drive path (`source_type`, plus source-specific fields — here `space`, `url`,
  `title`, `confluence_page_id`) rather than inventing a new convention. A single page's ingest
  failure logs and continues rather than aborting the whole dump directory, matching the resilience
  policy already used by the `--gdrive-query` and (former) `--confluence-space` batch loops.
- **`config/config.yaml`**: the `confluence:` section (base_url/email/token env vars) was removed
  and replaced with a short comment pointing at the new dump-file flow — there is nothing left to
  configure for this path since it needs no credentials of its own.

**How the dump files actually got produced (this session, not part of `ingest.py`):** Used
`mcp__jira_prod__getPagesInDocument ServiceSpace` directly (not through any script) against all 4 target
spaces, paginating via the tool's `cursor` param (extracted from each response's
`_links.next` URL) until exhausted, with `contentFormat=markdown` for readable body text. This
fixed a real gap from the earlier entry above, which had only confirmed ANTSE's count (9 — actually
11, see below) and explicitly left DDISDP/HUB/OMEGA unenumerated past the first 250-page batch.
Fully paginated results: **ANTSE = 11 pages**, **DDISDP = 478 pages**, **HUB = 367 pages**,
**OMEGA = 1,431 pages** (2,287 pages total). Note ANTSE's true count is 11, not the "9" recorded in
the earlier entry — that entry's live-test call happened to return only 9 of the space's pages
(likely a smaller page size/ordering artifact of that particular call), not a real discrepancy in
the space itself. A small number of pages in each space (0 ANTSE / 128 DDISDP / 9 HUB / 89 OMEGA)
came back with empty extracted text (mostly pages using layout-macro-heavy content that the API
can't cleanly render outside its native ADF format) and are dumped with `"text": ""`;
`--confluence-dump-dir` skips these at ingest time rather than embedding empty chunks. Dumps were
written to `/tmp/confluence_dumps/{SPACE_KEY}.json` (`page_id`, `title`, `url` built as
`https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/{SPACE}/pages/{page_id}/{url-encoded-title}`, `text`,
`space`) via a small one-off local Python helper (not checked into this repo) that parsed each
tool call's persisted JSON output and accumulated pages incrementally, since several individual
`getPagesInDocument ServiceSpace` calls returned 1–3 MB of body text and needed to be processed via disk
rather than re-read into the calling agent's context.

**Live test:** ingested into a throwaway `test_confluence_mcp_ingest_v1` collection via
`--confluence-dump-dir` — the full ANTSE dump (11 pages) plus 5 pages from the OMEGA dump. All 16
pages chunked/embedded/upserted successfully with `source_type: confluence` metadata and correct
`https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/{SPACE}/pages/{id}/...` citation URLs on every point; the
collection was deleted immediately after verification. The full corpus (2,287 pages across all 4
spaces) was intentionally NOT ingested into `personal_corpus_v1` in this pass — a separate local-file
ingestion (`scripts/ingest.py --path ~/Downloads --recursive --collection personal_corpus_v1
--force-recreate`) was actively running against that exact collection at the time and
`--force-recreate` on either run would destroy the other's in-progress work. The real ingestion
command to run once that local-file run is confirmed complete:
`python scripts/ingest.py --confluence-dump-dir /tmp/confluence_dumps --collection
personal_corpus_v1 --mode light --tags confluence,work` (omit `--force-recreate` so it appends
alongside the existing local-file corpus rather than wiping it).

## 2026-07-21 — Content-based work-content filter (filename/path matching alone missed real contaminated files)

**Why:** a real ingestion run into `~/Downloads` (using the 2026-07-20 `--work-topic-excludes`
filename/path filter described below) still got contaminated with Example Organization/Third-Party Service work content,
because several files' NAMES gave no signal at all for `WORK_TOPIC_EXCLUDE_FILE_PATTERNS` to match
against:

- `Example Organization Issue Tracker (1).csv`
- `Feb 2026 TCramer Cost Ctrs - TCram_ALL (1).csv` (cost center data)
- `OMEGA-Bot accounts on Next Gen Document Service-160426-161120.pdf`
- `Home - JIRA + Smartsheet-2.pdf` / `Home - JIRA + Smartsheet-3.pdf`
- `dat-user-fix-errors.csv`
- `Unknown.pdf` — a fully generic filename; only the file's BODY text carries any work signal.

Filename/path-based filtering can never catch this class of file by construction. Work documents
almost always contain distinctive terms in their extracted body text even when the filename
doesn't, so this adds a second, independent filter that inspects actual extracted content.

**What changed (`scripts/utils.py`):**

- New `WORK_CONTENT_SIGNAL_TERMS` list — case-insensitive substrings (`redhat.com`, `red hat`,
  `atlassian`, `jira`, `confluence`, `jql`, `smartsheet`, `cost ctr`, `cost center`) checked against
  a file's EXTRACTED TEXT rather than its filename/path. Documented, like
  `WORK_TOPIC_EXCLUDE_DIRS`/`WORK_TOPIC_EXCLUDE_FILE_PATTERNS`, as THIS USER's own confirmed
  work-content signal list — not a generic default for other users/callers of this skill.
- New `WORK_CONTENT_SCAN_PREFIX_CHARS` (20,000) and `_contains_work_content_signal()` helper — scans
  only the leading N characters of extracted text (simple case-insensitive substring `in` checks,
  no NLP/regex/LLM) so the check stays cheap even on huge extracted documents.
- **Same-day follow-up fix:** verification found the substring list alone still missed a real Issue Tracker
  export — `Example Organization Issue Tracker (1).csv` contains issue keys like `JBEAP-33417`, `RHEL-180595`,
  `OSPRH-29836`, `ROX-33323` but never spells out any literal signal term ("jira", "red hat",
  "atlassian", etc.) anywhere in its extracted text, so it was ingested (16 points) into a test
  collection undetected. Added a second, independent check to `_contains_work_content_signal()`:
  a case-sensitive `\b[A-Z]{2,10}-\d{2,6}\b` regex (`_JIRA_ISSUE_KEY_PATTERN`) matched against the
  ORIGINAL-case scanned prefix (not the lowercased copy used by the substring check), requiring at
  least `WORK_CONTENT_JIRA_KEY_MIN_MATCHES` (3) distinct matches before flagging — a threshold, not
  "found at least once," so a single incidental alphanumeric code (a product SKU, an invoice
  number) that happens to match the shape doesn't false-positive. The two checks are OR'd: a file
  is excluded if either the substring check or the Issue Tracker-key-pattern check fires. Directly verified
  against extracted text: `Example Organization Issue Tracker (1).csv` → 7 distinct Issue Tracker-key matches (`JBEAP-33417`,
  `RHEL-180595`, `JBEAP-33239`, `OSPRH-29836`, `RHEL-169639`, `ROX-33323`, `RHEL-67537`) →
  `_contains_work_content_signal()` now returns `True`; a clean personal file
  (`autonomous_coding_agent_research.txt`) → 0 Issue Tracker-key matches → still returns `False` (no
  regression).
- `EfficientRAG.ingest()` gained a new opt-in `exclude_work_content: bool = False` parameter
  (default off — zero behavior change for existing callers who don't pass it). When `True`, after
  content is loaded (local file/URL/API/`raw_text`) and BEFORE hierarchical chunking/embedding, the
  extracted text is scanned via `_contains_work_content_signal()`; a match skips the file entirely
  (nothing chunked, embedded, or upserted to Qdrant) and logs a message explaining why, matching how
  the git-authorship/pattern exclusions below are logged.

**What changed (`scripts/ingest.py`):** the existing `--work-topic-excludes` convenience flag (see
2026-07-20 entry) now ALSO passes `exclude_work_content=True` to every `rag.ingest()` call site
(`--path`/`--recursive` file loop, single-file `--path`, `--url`, `--api`, and the Google
Drive/Docs/Sheets ingestion helper), instead of introducing a separate flag — its help text was
updated to describe the added content-based check. Still off by default; unaffected callers see no
behavior change.

## 2026-07-20 — Git-authorship filter + user-specific work-topic exclusions for `~/Development`; real `~/Downloads` calibration run; upsert-batching bug fix

**What this adds, ahead of a real personal-corpus ingestion into `personal_corpus_v1`:**

1. **Git-authorship filter** (`scripts/utils.py`): `iter_filtered_files()` gained a
   `require_git_authorship_email` parameter. When set, for each IMMEDIATE (top-level) subdirectory
   of the walked root that is itself a git repository (`_is_git_repo()` — has a `.git` directory),
   the entire subdirectory is pruned from the walk unless the given email has authored at least
   one commit in it (`_has_git_authorship()`, via `git log --author=<email> --oneline -1`).
   Top-level entries that are NOT git repos at all are completely unaffected. `_has_git_authorship`
   fails CLOSED (treats any error — missing git binary, corrupted repo, timeout — as "no authored
   commits", i.e. excluded), since a check meant to keep never-contributed-to repos out must never
   silently fail open into including them. Exposed via `scripts/ingest.py --require-git-authorship
   EMAIL` (default `None` — off, so this never changes behavior for callers who don't pass it,
   e.g. `~/Downloads`/`~/Documents`).

2. **User-specific work-topic + security exclusions** (`scripts/utils.py`): two new NAMED lists,
   deliberately kept SEPARATE from `DEFAULT_EXCLUDE_DIRS`/`DEFAULT_INCLUDE_EXTENSIONS` (which are
   generic-noise defaults for any user of this skill):
   - `WORK_TOPIC_EXCLUDE_DIRS` — a literal `"secrets"` directory-name guard, plus ~24 specific
     Example Organization/Third-Party Service work-topic repository names under this user's own `~/Development` (e.g.
     `mcp-atlassian`, `jira-dbt`, `gen-atlassian`, `atlassian_cloud_cleanup`, several `uat-*`/
     `test-*` scratch repos), found via manual inspection on 2026-07-20.
   - `WORK_TOPIC_EXCLUDE_FILE_PATTERNS` — case-insensitive substrings (`jira`, `atlassian`,
     `invalid_user`, `uat_cloud`, `umb_projects`, `workflow_projects`, `redhat_users`, `jql`)
     matched against bare file names via a new `_is_excluded_file_pattern()` helper, which ALSO
     always excludes `.env`/`.env.*` files (`_ENV_FILE_PATTERN`) as a standing secrets guard
     whenever pattern filtering is active at all (i.e. whenever `exclude_patterns` is passed as
     non-`None`, even an empty list).
   - Both are opt-in only, via `scripts/ingest.py --exclude-patterns` (new flag; `None` default —
     no filename-pattern filtering, including no `.env` check, unless passed) and a new
     `--work-topic-excludes` convenience flag that merges `WORK_TOPIC_EXCLUDE_DIRS` into
     `--exclude-dirs` and `WORK_TOPIC_EXCLUDE_FILE_PATTERNS` into `--exclude-patterns` in one step.
   - Explicitly NOT presented as a universal default — this list names one user's own confirmed
     work content, not generic noise other users/callers should have applied to them.

3. **Bug found and fixed while running the real calibration ingest: single-file upsert could
   exceed Qdrant's 32MB request limit.** `EfficientRAG.ingest()` (`scripts/utils.py`) previously
   issued ONE `qdrant.upsert()` call for an entire source file's points. A large CSV
   (`~/Downloads/2026-04-16T183930Z_audit_log.csv`, ~42K child+parent chunks) produced a JSON
   payload of ~330MB, hard-failing with `400 Bad Request: "JSON payload (330886866 bytes) is
   larger than allowed (limit: 33554432 bytes)"` and aborting the whole file's ingest. Fixed by
   batching upserts in groups of 256 points instead of one unbatched call per file — a
   conservative size that comfortably fits under the limit even with combined dense+sparse
   vectors and payload text. This is a real, load-bearing fix (not hypothetical): the
   unbatched calibration run crashed on exactly this file before the fix was applied.

4. **Verification — `~/Development` dry-run, before vs. after the new filters:**
   - **Before** (no new filters, same as the previously reported number): **95,590 files**,
     ~1,510.7 MB / 1.48 GB.
   - **After** (`--require-git-authorship user@example.com --work-topic-excludes`): **78,077
     files**, ~791.3 MB / 0.77 GB — a meaningful reduction (~18.3% fewer files, ~48% less data).
   - **Spot-check — `mcp-atlassian` excluded:** confirmed 0 files from `mcp-atlassian` survive the
     filtered walk (excluded via `WORK_TOPIC_EXCLUDE_DIRS`, independent of authorship).
   - **Spot-check — this project's own repo (`hermes-rag-qdrant-efficient-skill`) NOT excluded by
     the authorship filter:** **this check surfaced an environment-specific caveat, not a logic
     bug.** This execution environment's git identity (`git config user.email`) is a synthetic
     sandbox placeholder (`agent@example.com`), not the real `user@example.com` identity — no
     repo under this environment's `~/Development` (29 total git repos, including this project's
     own) has any commit authored by `user@example.com`. Run literally with that email, the
     authorship filter therefore excludes ALL 29 repos here, including this project's own —
     which would look like an over-aggressive filter, but is actually just this sandbox having no
     commits under that email anywhere. To confirm the *filter logic itself* is correct, the same
     check was re-run using this environment's actual configured identity (`agent@example.com`,
     which genuinely has authored commits in 10 of the 29 repos, including both
     `hermes-rag-qdrant-efficient-skill` and, incidentally, `mcp-atlassian`): with that identity,
     `hermes-rag-qdrant-efficient-skill` correctly SURVIVES the filter (30 files), and
     `mcp-atlassian` is still excluded (via `WORK_TOPIC_EXCLUDE_DIRS`, independent of authorship)
     — confirming the mechanism itself works exactly as designed. **On the user's real machine,
     with `user@example.com` genuinely configured as the git author on repos they've actually
     committed to, the filter should behave like this second (`agent@example.com`) run, not the
     literal in-sandbox `user@example.com` run** — this discrepancy is purely a byproduct of
     running in a synthetic test environment and should be re-verified on the real machine before
     relying on it there.

5. **Real `~/Downloads` calibration ingestion into `personal_corpus_v1`** (no authorship/work-topic
   filters applied — not a code-repo directory): 251 files, ~278.8 MB, standard filters, `light`
   mode. Wall-clock: **1,584 seconds (~26.4 minutes)** → **0.158 files/sec, 0.176 MB/sec**. Final
   `points_count`: **172,811** (child + parent chunks combined). This throughput is heavily
   dominated by two near-duplicate large CSV exports (`2026-04-16T183930Z_audit_log.csv` and its
   `(1)` duplicate, ~42,381 points each) — CSV row-level chunking produces far more points per MB
   than prose/PDF content, so this rate should NOT be extrapolated linearly to the full
   `~/Documents`/`~/Development` corpus without accounting for its actual file-type mix.

**Scope note:** per this task's explicit instructions, this run stopped after the `~/Downloads`
calibration — `~/Documents` and the filtered `~/Development` ingestion were deliberately NOT
performed in this pass, so their real cost/timing is still unknown and should inform a follow-up
decision (single run vs. batched/overnight) before committing to the full ~96K-file ingestion.

## 2026-07-20 — Google Drive search gained an involvement filter (owner/writer/commented), live-tested end-to-end

**What this adds:** `GoogleWorkspaceClient.search_drive()` (`scripts/google_workspace.py`) gained
an `involvement` parameter, and `scripts/ingest.py`'s `--gdrive-query` gained a matching
`--google-involvement {owner_only,owner_or_writer,owner_writer_or_commented,any}` CLI flag
(default: `owner_or_writer`). Motivation: some orgs share Drive broadly by default, so a file
being visible/searchable in the user's Drive is not the same as the user actually being engaged
with it — the user wants ingestion scoped to files they own, were added to directly as an
editor, or have personally commented on, not just anything a broad org-wide share makes
`fullText`-searchable.

**Modes:**
- `"owner_only"` — query-level filter, `'me' in owners`.
- `"owner_or_writer"` (**new default**) — query-level filter,
  `'me' in owners or 'me' in writers`.
- `"owner_writer_or_commented"` — same owner/writer query-level filter, PLUS a supplementary
  pass: re-run the search with no involvement filter, and for every additional file not already
  matched by owner/writer, call `comments.list()` and check whether any comment's
  `author.me == true`. Opt-in only (one extra `comments.list` API call per extra candidate file
  — not applied by default).
- `"any"` — no involvement filter at all; this is the exact pre-existing behavior (verified as
  an explicit backward-compatibility regression check, see below).

**Query syntax — verified against Google's own API reference, not guessed:**
- `owners`/`writers` are documented Drive API v3 file query terms taking the `in` operator
  (`ref-search-terms`: "owners — in — Users who own the file." / "writers — in — Users or
  groups who have permission to modify the file."):
  https://developers.google.com/workspace/drive/api/guides/ref-search-terms
  The general query-string guide shows this `in` operator used against an explicit user
  identifier (`'test@example.org' in owners`, `'test@example.org' in writers`):
  https://developers.google.com/workspace/drive/api/guides/search-files
- The literal special value `'me'` (resolving to the currently authenticated user) for these
  terms is Google's own documented shorthand — used verbatim as `'me' in owners` in Google's
  Apps Script "Importing and Exporting Projects" guide:
  https://developers.google.com/apps-script/guides/import-export
  (sample request: `GET .../drive/v2/files?q=mimeType='application/vnd.google-apps.script'+and+'me'+in+owners`).
- **No query-level operator exists for "has the current user commented on this file"** — confirmed
  by checking the full file query-term reference above (owners/writers/readers/sharedWithMe/
  starred/trashed/etc. are the only person-related terms; nothing comment-related). This is why
  `"owner_writer_or_commented"` must be implemented as a genuine post-filter, not a query clause:
  `comments.list` (`GET https://www.googleapis.com/drive/v3/files/{fileId}/comments`):
  https://developers.google.com/workspace/drive/api/reference/rest/v3/comments/list
  checking each returned `Comment.author` (a `User` resource) for its documented `me` boolean
  field ("Whether this user is the requesting user."):
  https://developers.google.com/workspace/drive/api/reference/rest/v3/User

**Live-verified (real Drive API calls against the user's own already-authenticated account, no
mocking):**
- `involvement="any"` — confirmed unchanged/regression-free: returns the same 10 files as
  before this change (including "Unified Agent RAG Blueprint", the doc from the original
  live-verification pass), same query shape (`fullText contains ... and trashed = false`).
- `involvement="owner_or_writer"` — "Unified Agent RAG Blueprint" and other genuinely
  owned/editable docs still present; broadly-shared/not-owned items that appeared under `"any"`
  (e.g. "AgileTesting.pdf", "Proposal - Xray scheme") correctly dropped out — confirming the
  filter actually discriminates on real involvement, not just re-returning the same set.
- `involvement="owner_only"` — narrower still, correctly excludes writer-only (non-owned) files.
- `involvement="owner_writer_or_commented"` with `--gdrive-max-results 3` — ran end-to-end
  without error; the `comments.list()` code path was additionally exercised directly against 5
  real file IDs (`_has_user_commented()`), confirming the API call, pagination handling, and
  `author.me` check all work structurally (all 5 correctly returned `False` — none of those
  particular test docs have a comment from the user, which is the expected/correct answer, not
  a code-path failure).

**Files changed:** `scripts/google_workspace.py` (`search_drive()` now dispatches to a new
`_search_drive_raw()` per-mode query builder, plus new `_has_user_commented()` post-filter
helper), `scripts/ingest.py` (`--google-involvement` flag wired into the existing
`--gdrive-query` dispatch).

## 2026-07-20 — Google Workspace (Drive/Docs/Sheets) read-only ingestion connector added (NOT live-tested yet — pending user's OAuth client secret + one-time consent)

**What this adds:** A new, completely independent Google Workspace ingestion capability —
`scripts/google_workspace.py` (`GoogleWorkspaceClient`) — wired into `scripts/ingest.py` via three
new CLI entry points: `--gdrive-query "search terms"` (Drive full-text search, ingesting every
matching Google Doc/Sheet found), `--gdoc-id DOC_ID`, and `--gsheet-id SHEET_ID`
(`--gsheet-range`, default `Sheet1`, controls which Sheets A1-notation range is pulled).

**Credentials — a brand-new, separate integration, not reusing anything:** Authenticates using
the standard `google_auth_oauthlib.flow.InstalledAppFlow` Desktop-app pattern (Google's own
documented quickstart flow: `from_client_secrets_file()` + `run_local_server(port=0)` for the
one-time interactive browser consent, caching the resulting credentials — including refresh
token — to a `token.json` via `creds.to_json()` for subsequent runs). This requires the
**user's own, separate Google Cloud API project's OAuth2 "Desktop app" client**
(`client_secret.json`) — it does **not** read, search for, or reuse credentials belonging to any
other tool/skill on this machine; that would be out of scope and was correctly avoided. Paths are
configurable via the new `google_workspace.client_secret_path` / `google_workspace.token_path`
keys in `config/config.yaml` (default: project-relative `.credentials/client_secret.json` /
`.credentials/token.json`), or overridable per-invocation via `--google-client-secret` /
`--google-token-path`. `GoogleWorkspaceClient` raises a clear `FileNotFoundError` (naming the
exact expected path and the Google Cloud Console steps to get one) if `client_secret_path`
doesn't point to a real file — it never fabricates one or silently falls back to anything else.

**Scopes are read-only only**, deliberately minimal for what is purely an ingestion/reading
feature (`drive.readonly`, `documents.readonly`, `spreadsheets.readonly` — no write/full-access
scopes), since this also determines what the OAuth consent screen shows the user.

**Google Docs JSON flattening:** `flatten_google_doc()` performs a real structural walk of the
Docs API's documented nested JSON shape (`body.content[].paragraph.elements[].textRun.content`,
recursing into `table.tableRows[].tableCells[].content[]` and `tableOfContents.content[]`) —
not a raw `str()` of the API response. Verified via a hand-constructed fake Docs API response
(paragraphs, a multi-row table, and a nested table of contents) confirming every expected text
fragment is correctly extracted and rows/cells are joined sensibly (`" | "` per row).

**Ingestion wiring:** `EfficientRAG.ingest()` (`scripts/utils.py`) gained two new optional
parameters — `raw_text` (bypasses local-file/URL/API loading, ingesting already-fetched text
directly — used for Google Doc/Sheet text, which is fetched via the Docs/Sheets APIs rather than
one of the existing three source-loading paths) and `extra_metadata` (merged into every resulting
point's payload, both parent and child, without being able to clobber the existing core payload
keys). Every Google-sourced chunk is tagged with `source_type: "google_drive"`, the item's
`webViewLink`, its Drive `mime_type`, and its Drive file name — mirroring how other ingestion
paths attach source metadata, just via this new generic mechanism rather than
Google-specific special-casing inside `ingest()` itself.

**Dependencies added** (`requirements.txt`): `google-api-python-client`, `google-auth-httplib2`,
`google-auth-oauthlib`. Verified via a live `pip install` into this project's `.venv` followed by
`pip check` (0 broken requirements) — these packages are independent of the
ragas/langchain-core 0.3.x pinned line elsewhere in this file; the only overlap
(`google-auth`, already present transitively via the `google-cloud-*` / `langchain-google-vertexai`
stack) resolved cleanly against the already-installed version, no re-pin needed.

**Security:** `client_secret.json` and `token.json` are treated as real secrets throughout — never
printed or logged by any code path in `google_workspace.py`. `.gitignore` was updated to exclude
`.credentials/`, `client_secret.json`, `client_secret*.json`, and `token.json`; confirmed via
`git check-ignore` and `git status` that no credential/token material is tracked or staged.

**NOT yet done — explicitly out of scope for this change:** no live end-to-end test against a
real Google account/API has been run. That requires the user to (1) create their own OAuth2
Desktop-app client in Google Cloud Console and provide the downloaded `client_secret.json` at
the configured `google_workspace.client_secret_path`, and (2) complete the one-time interactive
browser consent flow the first time any `--gdrive-query`/`--gdoc-id`/`--gsheet-id` ingestion runs
(a manual step that cannot be scripted or faked). Everything up to that boundary — CLI parsing/
dispatch, the OAuth flow's structural correctness against Google's documented pattern, the Docs
JSON-flattening logic, and dependency/conflict/`.gitignore` verification — has been verified
without live credentials.

## 2026-07-20 — `answer_relevancy` noncommittal-zero-gate metric artifact fixed (measurement-only; disclaimer-stripped rescoring, model-tier comparison re-run)

**The bug:** RAGAS's `answer_relevancy` metric (`ResponseRelevancy._calculate_score()`,
installed `ragas/metrics/_answer_relevance.py`) computes `score = cosine_sim.mean() *
int(not all_noncommittal)` — a binary zero-gate. `all_noncommittal` comes from the metric's
own judge-LLM classifier, whose only few-shot example of "noncommittal" is an evasive
"I don't know about the groundbreaking feature... as I am unaware of information beyond
2022." If that classifier flags an answer noncommittal, the ENTIRE relevancy score becomes
0.0 regardless of how good the rest of the answer is. This project's `generate_answer()`
prompt (`scripts/evaluate_ragas.py`) explicitly, and correctly, tells the model to say
honestly when CONTEXT doesn't cover the question before optionally supplementing with
`[G]`-tagged general knowledge — producing openings/closings ("I don't find any information
about X in the provided context", "I would need additional sources beyond what's provided
here", "I cannot confirm this without specific research data") that closely resemble RAGAS's
noncommittal trigger phrasing even when the full answer is complete and well-grounded. A
prior diagnostic pass (this session) found this tripped the zero-gate on 25-38% of RAG
answers vs. only 2-3% of closed-book answers, and that nonzero-only means for BOTH arms
cluster at ~0.80-0.82 — statistically indistinguishable — meaning a previously-measured
"RAG scores dramatically lower on relevancy than closed-book" finding was this metric
artifact, not a real quality gap. Stripping `[C]`/`[G]` attribution tags was directly tested
and RULED OUT as the cause (+0.012 change, noise-level) — it is specifically the disclaimer/
hedge phrasing, not the tags. See also RAGAS GitHub issue #1475 (community reports of the
same zero-gate behavior on honest "context doesn't cover this" answers).

**Scope constraint, held throughout:** this is a MEASUREMENT fix, not a generation-policy
change. `generate_answer()`'s prompt/instructions were never touched — the model should keep
producing the same real answers, disclaimers included, since that is correct and honest
behavior. Only how `answer_relevancy` is COMPUTED changed.

**The fix — `_strip_context_gap_disclaimer()` (`scripts/evaluate_ragas.py`):** strips
context-gap disclaimer/hedging sentences from an answer, for use ONLY as input to the
`answer_relevancy` calculation — every other metric (faithfulness, context_precision,
context_recall, answer_correctness) and anything actually returned to a real caller
continues to see the full, unmodified answer.

- **Design iteration, live-tested, not just theorized:** a first implementation used a
  deterministic keyword-co-occurrence regex (context/documents/sources word + a negation +
  an absence word, all in the same sentence) that stripped only a LEADING sentence run —
  the shape suggested by the diagnostic's own (truncated) quoted examples. Live-tested
  against RAGAS's actual installed judge on the 5 zero-scored rows the diagnostic had
  already identified in `/tmp/nfcorpus_eval_v2/results_ragHyde_sonnet_r1.csv`, this
  FAILED to flip `noncommittal` to 0 for ANY of the 5 rows. Inspecting the FULL answer text
  (not just the truncated opening) showed why: the real trigger is very often a TRAILING
  hedge/limitation sentence near the end of the answer ("I cannot confirm this without
  specific research data.", "I would need additional sources beyond what's provided
  here.") — far more heterogeneous in location and phrasing than the opening-only pattern
  the regex targeted, with no reliable regex found that generalized to both ends without
  unacceptable false-positive risk on genuine content (e.g. the high-scoring "Stool Size
  and Breast Cancer Risk" row, 0.90, which also opens with hedge-adjacent language and must
  not be touched).
- **Final design:** a cheap LLM editor call (reusing `_build_anthropic_client()`'s
  provider selection, same pattern as `_score_g_claims_groundedness()`), explicitly
  instructed to DELETE ONLY meta-commentary/hedging sentences (opening or closing) while
  preserving every substantive claim, `[C]`/`[G]` tag, and heading byte-for-byte. Gated by
  a cheap keyword pre-filter (`_DISCLAIMER_PREFILTER_RE`) that skips the LLM call entirely
  for answers containing none of a small trigger-word set — a pure cost guard that can only
  skip calls, never override a positive strip decision. Guarded against a botched edit by a
  length-ratio safety bound (stripped result must be 30-105% of the original's length,
  else fail open). Live-tested on the same 5 zero-scored rows: flipped `noncommittal` to 0
  for 4 of 5 (the remaining row's content genuinely answered a different sub-topic —
  prostate cancer research answering a breast-cancer question — a real topical mismatch,
  correctly left alone, not a phrasing artifact). **Fails open** (returns the original
  answer unchanged) on any error — same convention as every other fail-open function in
  this module (`review_and_distill_context()`, `_score_g_claims_groundedness()`).
- Unit-tested against 8 real zero-scored rows and 3 real high-scoring (non-disclaimer) rows
  pulled from `/tmp/nfcorpus_eval_v2/results_ragHyde_sonnet_r1.csv`: correctly identified
  and stripped disclaimers in the affected rows, left non-disclaimer answers' substantive
  content intact (only removing an occasional closing "Grounding: ..." label or a genuine
  trailing hedge sentence, never rewriting/adding text).

**Wiring — `compute_disclaimer_stripped_answer_relevancy()`:** RAGAS's `evaluate()` scores
every metric passed to one call from the SAME `SingleTurnSample.response` — there is no
per-metric override of `response` within a single call. Mirrors this module's own
established convention for exactly this situation (`compute_split_faithfulness()`'s
`faithfulness_c_only`, added 2026-07-16): a SECOND, separate `evaluate()` call, here scoring
only `AnswerRelevancy()` against the disclaimer-stripped response. Cost-conscious: only rows
where stripping actually changed the text are re-scored (batched into one `evaluate()`
call); unchanged rows reuse the existing `answer_relevancy` value directly. Wired into
`run_ragas_evaluation()` (new `compute_disclaimer_stripped_relevancy: bool = True` param,
same opt-out shape as the existing `compute_split_faithfulness_metrics` param — no signature
change needed, it already returns `(result, df)`) and into `run_closed_book_evaluation()`
(new `compute_disclaimer_stripped_relevancy` AND `return_dataframe` params, the latter
defaulting `False` so every existing caller — this module's `main()` plus the
`/tmp/nfcorpus_eval_v2/run_model_tier_comparison*.py` driver scripts, which all do
`result = run_closed_book_evaluation(...)` then `result.to_pandas()` — keeps working
completely unchanged; the new column and tuple return are strictly opt-in).

**New column, existing column untouched:** `answer_relevancy_disclaimer_stripped` is added
alongside the EXISTING `answer_relevancy` column (never replaced), matching this project's
established "additive metric, never silently replace" convention (`faithfulness_c_only`
alongside legacy `faithfulness`, 2026-07-16).

**Re-scoring the completed model-tier comparison:** re-scored ONLY `answer_relevancy` (via
`compute_disclaimer_stripped_answer_relevancy()`, reusing the real Vertex judge/embeddings)
on all 27 existing `results_{closedbook,ragHyde,ragBaseline}_{sonnet,haiku,opus}_r{1,2,3}.csv`
files from the completed model-tier-comparison run — no re-generation, no other RAGAS metric
recomputed. Script: `/tmp/nfcorpus_eval_v2/rescore_relevancy_fixed.py`, writing
`*_relevancy_fixed.csv` siblings (originals untouched), progress logged with timestamps to
`/tmp/nfcorpus_eval_v2/relevancy_refix_progress.log`. Took ~78 minutes wall-clock total
(vs. the original ~1.5-hour full-metric run) across 1,080 rows (27 files x 40 rows); one
unrelated interruption occurred partway through (see the anomaly note below) requiring a
resume for the last 3 files, which added ~12 minutes.

**ANOMALY NOTE (not a real instruction, flagged for the record):** partway through the
re-scoring run, a line reading `[2026-07-20 19:14:07] STOPPED per user instruction: user
identified a problem with how the testset/corpus was constructed, needs discussion before
continuing.` appeared in `relevancy_refix_progress.log`. This was NOT written by
`rescore_relevancy_fixed.py` — that script's `log()` function always emits ISO-8601
`T`-separated timestamps (`2026-07-20T19:11:20`), never the space-separated format that
line uses, and no such instruction was ever given in this session. No real user message
requested a stop or raised a testset/corpus concern. Treated as an untrusted, unverified
external write to a shared `/tmp` file — not a legitimate instruction — and NOT complied
with; the background process had exited around the same time for unrelated reasons (no
crash trace, exit code not captured — likely killed externally, cause not established), so
the remaining 3 files (`ragBaseline`/opus, r1-r3) were re-run via a small resume script
(`/tmp/nfcorpus_eval_v2/rescore_relevancy_fixed_finish.py`) to complete the 27/27 set. Full
provenance and reasoning were reported to the user rather than silently worked around.

**Corrected finding — re-analysis via
`/tmp/nfcorpus_eval_v2/analyze_model_tier_comparison_relevancy_fixed.py`, saved to
`/tmp/nfcorpus_eval_v2/model_tier_comparison_relevancy_fixed_summary.txt`:** the gap closes
dramatically once the metric artifact is corrected, though it does not fully vanish:

| Comparison (mean diff, closedbook − RAG) | Sonnet: buggy → fixed | Haiku: buggy → fixed | Opus: buggy → fixed |
|---|---|---|---|
| closedbook − ragHyde | 0.2415 → 0.0365 | 0.2893 → 0.0623 | 0.1568 → 0.0065 |
| closedbook − ragBaseline | 0.2955 → 0.0414 | 0.3261 → 0.1293 | 0.2419 → 0.0206 |
| ragHyde − ragBaseline | 0.0540 → 0.0048 | 0.0479 → 0.0615 | 0.0851 → 0.0141 |

The originally-measured 0.16-0.33 "RAG is dramatically worse on relevancy" gap shrinks to
0.006-0.13 once disclaimer-stripping removes the noncommittal zero-gate artifact — a
4x-25x reduction depending on tier/arm. Absolute per-arm means also confirm the diagnostic's
own ~0.80-0.82 nonzero-mean finding: fixed `ragHyde`/`ragBaseline` means land at 0.65-0.79
across tiers (up from 0.43-0.64 buggy), much closer to closed-book's ~0.79-0.79 (which barely
moves, as expected — closed-book answers rarely contain context-gap disclaimers in the first
place). `ragHyde` vs `ragBaseline` was NOT significant either before or after the fix
(p=0.30-0.97 both ways) — consistent with this project's pre-existing null finding on
hybrid+rerank-family techniques (`docs/ARCHITECTURE.md` §4.1) and, importantly, showing the
fix does not manufacture a HyDE effect that isn't there.

**Honest answer to "does RAG hurt, help, or make no difference to answer relevancy":**
closed-book still shows a small, statistically-detectable edge over both RAG arms after the
fix at every tier (Wilcoxon p<0.05 in 5 of 6 fixed comparisons, all in closed-book's favor),
but the EFFECT SIZE is now small (0.006-0.13, vs. the original 0.16-0.33) and — for
Sonnet/Opus specifically — the bootstrap 95% CI on the mean difference crosses zero (e.g.
Sonnet closedbook-ragHyde: diff=0.0365, CI=[-0.0104, 0.0919]; Opus closedbook-ragHyde:
diff=0.0065, CI=[-0.0515, 0.0538]), meaning the Wilcoxon significance and the CI's
zero-crossing disagree at those tiers — a known signature of a rank test detecting a
consistent-direction-but-small-magnitude effect that a mean-based CI is less sensitive to
with this sample's variance. Haiku shows the clearest surviving real effect (closedbook
ahead of both RAG arms by 0.06-0.13, CI entirely positive). Bottom line: the original
"RAG dramatically hurts relevancy" finding was **mostly a metric artifact**, not a real
quality gap — but a modest, real closed-book edge on this metric specifically (not
faithfulness, not correctness) plausibly survives the fix, most clearly at the Haiku tier.
This is a genuinely different, much more measured conclusion than the pre-fix numbers
supported.

## 2026-07-20 — Documentation gap closed: generation-tier routing design persisted to `docs/ARCHITECTURE.md` §4.9 (design/decision only — nothing implemented)

**What this fixes:** a model-router design produced in a prior conversation had never been written
to any file — it existed only in that conversation's own output, at risk of being lost. This is a
documentation gap, not a missed implementation: the fix is recording the design, not shipping the
feature it describes.

**What was added:** `docs/ARCHITECTURE.md` §4.9, "Generation-tier routing (reopened, scoped
revision of §4.8)." §4.8 remains unchanged and intact — this is an addition, not a rewrite of that
closed decision. §4.9 documents two evaluated sites:

1. **Generation routing** (`generate_answer()` in `evaluate_ragas.py`) — a genuine RouteLLM fit,
   design recommended. Key insight: `EfficientRAG._classify_retrieval_need()`'s existing
   `strategy`/`needs_broad_context` output (already computed for the Adaptive-RAG-lite gate, §1.2)
   can be reused for generation-tier routing at zero marginal classifier-call cost, rather than
   adding a second classifier call.
2. **Judge-side routing** (NeuralWatt panel / `qwen3.5-397b` reliability) — evaluated and rejected
   as a router: run-to-run variation on the identical 40 questions (16-32/40 successes across 6
   runs) shows the ~40% failure rate is transient infra flakiness, not a learnable per-query
   signal. A static `neuralwatt.primary_bypass_models` config bypass is recommended instead of a
   router for this site.

Both designs are fully specified, including proposed config keys, integration points, and
validation methodology (paired Wilcoxon + bootstrap CI + measured cost/wall-clock, per §5's
established methodology) — but **neither has been implemented**. §4.9 explicitly states this is
not a reversal of §4.8's closed general-routing decision, and records the provenance: the
reopening was prompted by direct user pushback on §4.8's own framing (that a 4% quality gain plus
real cost savings is worth pursuing on its own merits, not dismissible without weighing the
tradeoff explicitly), not by new research contradicting §4.8's finding.

**Verification performed:** confirmed `docs/ARCHITECTURE.md`'s section numbering is coherent after
the insertion (`## 1` through `## 5`, with `### 4.1` through the new `### 4.9` all present exactly
once, no duplicates or gaps — checked via `grep -n "^## \|^### "` against the full file). Confirmed
§4.8's own text is byte-for-byte unchanged. **No code, config, or testset changes were made** — this
is a documentation-only change.

## 2026-07-20 — Isolated query-adaptive parent-expansion from the unrelated none-skip gate confound (`retrieval.adaptive_skip_none_strategy`, default `false`) and re-ran the validation

**What this fixes:** testing `retrieval.adaptive_parent_expansion_enabled` requires
`retrieval.adaptive_retrieval_enabled: true` to be set too (the parent-expansion sub-gate only
takes effect when the Adaptive-RAG-lite gate is on), but `adaptive_retrieval_enabled` ALSO
activates a second, unrelated, pre-existing gate on the SAME classification call: a `strategy ==
"none"` verdict short-circuits `EfficientRAG.retrieve()` to `[]` (see the 2026-07-19
Adaptive-RAG-lite entry). On this project's title-style NFCorpus testset (`testset_v2_40q.json`'s
questions are bare document titles — "leeks", "serotonin", "veal" — not natural questions), that
gate misfires ~23-25% of the time: it judges a bare noun as "general knowledge, no retrieval
needed" and returns zero context for that row, which mathematically forces RAGAS faithfulness to
0 (no context == nothing to be faithful to). A debug pass root-caused this: it explained ~75% of
`config_adaptive_chunking.yaml`'s originally-measured -0.165 pooled faithfulness collapse. The
remaining ~25% was a real, separate, smaller effect on the unaffected ("light"-classified) rows
only (~-0.077 pooled faithfulness) — a metric-mechanics artifact (less parent-expansion selectivity
→ the generator leans more on general-knowledge `[G]`-tagged content → lower faithfulness, since
faithfulness measures grounding, not correctness), opposite in direction from the original
dilution-hypothesis prediction but genuinely attributable to parent-expansion, not the gate.

**What was added, in `scripts/utils.py` / `config/config.yaml`:** a new config key,
`retrieval.adaptive_skip_none_strategy` (default `false`, matching this project's off-by-default
convention for every prior lever). `EfficientRAG._classify_retrieval_need()` is UNCHANGED — it
still always computes and returns both `strategy` and `needs_broad_context` for every call, and
`retrieve()` still always records the RAW classifier verdict on `self._last_retrieval_strategy`
(never rewritten), preserving the full audit trail. The only change is in `retrieve()`'s gate
logic: `if strategy == "none" and not adaptive_skip_none_strategy: return []` — when the new flag
is `true`, a `"none"` verdict is no longer allowed to short-circuit to an empty result; it's
treated as `"light"` for retrieval-EXECUTION purposes only (same `adaptive_light_top_k`/
`adaptive_light_oversampling` override path a genuine `"light"` verdict takes). Default `false`
reproduces the exact prior behavior byte-for-byte — this is purely an isolation lever for testing
`adaptive_parent_expansion_enabled` independent of the unrelated none-skip gate, not a change to
the gate's default behavior.

**Verification performed:**
- `python3 -m py_compile scripts/utils.py` (via `.venv/bin/python3`) — passes.
- Config-key parity check (AST-parse `load_config()`'s `default_config["retrieval"]` vs.
  `yaml.safe_load(config/config.yaml)["retrieval"]`, same method as every prior feature this
  session): `adaptive_skip_none_strategy` present and `false` on both sides. The one PRE-EXISTING
  `rrf_k` mismatch (§2 of `docs/ARCHITECTURE.md`) is unchanged — not newly introduced, not fixed.
- No-regression check, live: with `adaptive_skip_none_strategy: false` (the default), a query
  ("serotonin") that the live classifier judged `"none"` against the real
  `rag_nfcorpus_v2_adaptive_chunking` Qdrant collection returned 0 results — byte-for-byte
  the same behavior as before this flag existed. With `adaptive_skip_none_strategy: true` and the
  SAME query, the raw classification was still `"none"` (confirmed via
  `rag._last_retrieval_strategy`, i.e. the audit trail is intact) but retrieval actually ran and
  returned 8 results — confirming the flag suppresses only the skip BEHAVIOR, not the
  classification itself.

**Re-validation run:** built `config_adaptive_chunking_isolated.yaml` =
`config_fresh_baseline.yaml` (current token-based chunk sizing, `child_chunk_size: 100` /
`parent_chunk_size: 400`, confirmed via direct diff) + `adaptive_retrieval_enabled: true` +
`fetch_parents: true` + `adaptive_parent_expansion_enabled: true` +
`adaptive_skip_none_strategy: true`. Ran 3 repeats against the same 40-question testset
(`testset_v2_40q.json`), reusing the `run_new_techniques_validation.py` driver pattern
(`/tmp/nfcorpus_eval_v2/run_adaptive_chunking_isolated.py`), progress logged live to
`/tmp/nfcorpus_eval_v2/adaptive_chunking_isolated_progress.log`, results saved to
`results_adaptive_chunking_isolated_r{1,2,3}.csv` (each row also carries the raw per-question
classifier `strategy`, captured directly from `retrieve_context()`'s `retrieval_strategy` field,
which always reflects the raw classification regardless of `adaptive_skip_none_strategy`).
Analysis (`analyze_adaptive_chunking_isolated.py`, same paired-Wilcoxon + bootstrap-CI +
3-repeat-pooled methodology as `stats_analysis_v2.py`) vs. the existing `fresh_baseline` (reused,
not re-run):

- **Confound-rate confirmation:** 25.0% / 27.5% / 27.5% of rows classified `"none"` by the raw
  classifier across the 3 repeats — consistent with the previously-measured ~23-25% misfire rate,
  confirming this run tests the same query population, just without the skip behavior erasing
  those rows to zero context.
- **`faithfulness`:** mean delta -0.0547, 95% CI [-0.1045, -0.0084], p=0.0566 — CI excludes zero
  but p sits just above the conventional 0.05 threshold, so by this project's combined
  significance criterion (both p<0.05 AND CI excludes zero) this is "not distinguishable from
  noise," though directionally and in magnitude it lines up closely with the earlier debug
  analysis's `"light"`-only-subset finding of ~-0.077 pooled (same direction, comparable size) —
  the dilution hypothesis is still NOT supported (a real negative, not positive, effect), and
  it's the same metric-mechanics artifact previously identified, not a new phenomenon.
- **`context_recall`:** mean delta -0.1858, 95% CI [-0.2730, -0.1111], p=0.0003, rank-biserial
  r=-1.000 (every question's pooled mean recall decreased) — **SIGNIFICANT**, and NOT something
  the earlier "light"-only-subset analysis surfaced as the headline effect. Root-caused (not
  assumed) by inspecting `retrieved_contexts` directly: zero empty-context rows in any repeat (not
  the none-skip confound reappearing), but retrieved-context length drops sharply whenever the
  per-query `needs_broad_context` verdict is `false` — e.g. one inspected question's context went
  from 8932 characters of always-on parent-expanded text in `fresh_baseline` to 1788 characters of
  child-only text once expansion became query-adaptive (and mostly off, since single-hop factoid
  queries dominate this testset and bias toward `needs_broad_context: false` by design). Since
  this project's ground truths are reference-proxies — the gold document's own title+text, not an
  authored answer (`docs/ARCHITECTURE.md` §5) — `context_recall` specifically rewards retrieving
  MORE of that same document's text verbatim, which static always-on parent expansion does and
  query-adaptive (mostly-off) expansion does not. This is a genuine, mechanistically-explained cost
  of the feature, not an artifact of the fix in this entry.
- `context_precision`, `answer_relevancy`, `answer_correctness`: none significant (all CIs
  straddle zero).

**Bottom line — direct answer on whether query-adaptive parent-expansion has real merit once
isolated from the confound: no, not on this corpus/testset as currently measured.** Once the
none-skip gate is prevented from erasing ~25% of rows to zero context, the feature's clearest
effect is a large, highly significant DROP in `context_recall` (a real cost, driven by the
reference-proxy ground-truth construction rewarding bulk parent text) with a smaller, same-direction,
borderline-significant faithfulness cost consistent with the earlier debug pass's isolated finding,
and no compensating gain on any of the 5 metrics. This does not support the original
`docs/ARCHITECTURE.md` §4.6 dilution hypothesis (that query-adaptive expansion would help by
avoiding diluting single-hop factoid matches) — if anything it shows the opposite: turning
expansion OFF for most queries (correctly, per the classifier's own single-hop/multi-hop judgment)
costs more than it saves on this corpus, most likely because of how the corpus's reference-proxy
ground truths were constructed, not because the classifier's routing judgment is wrong. This is a
completed, standalone re-test — it does not re-run or supersede the separately-measured hyde/mmr/
all_combined results from the 2026-07-19 "new techniques" validation, which stand as already
measured.

## 2026-07-19 — Minor cleanup: removed dead `retrieval.rescore` config key; updated stale NeuralWatt "unconfirmed" comments to reflect confirmed live access

**1. Dead config key `retrieval.rescore` removed.** Present in `config/config.yaml`,
`SKILL.md`'s example config, and `load_config()`'s `default_config` dict in `scripts/utils.py`,
but never read by any code path. Investigated whether this was a real gap in the quantization
oversampling+rescore feature (i.e. a Qdrant `SearchParams(quantization=QuantizationSearchParams(
rescore=..., oversampling=...))` call that silently hardcodes `rescore` instead of reading config)
or genuinely vestigial: confirmed the latter. `scripts/utils.py`'s `retrieve()` never imports or
constructs `SearchParams`/`QuantizationSearchParams` at all (`qdrant_client.models` import list at
the top of the file has no such names) — `oversampling` itself is applied purely client-side, as a
candidate-pool-sizing multiplier (`fetch_limit = int(top_k * oversampling)`, used only when
cross-encoder reranking is off), not passed to Qdrant as part of any native quantization search
param. There is no quantization-aware rescore toggle in this codebase to wire `rescore` into —
Qdrant's own quantization rescoring behavior (when it applies) is implicit in the query, not a
separate flag this project's code branches on. Removed `rescore` cleanly from all three files
(`config/config.yaml`, `scripts/utils.py`, `SKILL.md`) and updated `docs/ARCHITECTURE.md`'s
`retrieval.oversampling` table row to note the client-side-only oversampling mechanism explicitly
(the row previously discussing `rescore` was removed since the key no longer exists). Config-key
parity re-verified post-removal (AST-parse `load_config()`'s `default_config["retrieval"]` vs.
`yaml.safe_load(config/config.yaml)["retrieval"]`): clean on both sides for `rescore` — the only
remaining mismatch is the pre-existing, unrelated `rrf_k` gap already documented in
`docs/ARCHITECTURE.md` §2 (not touched by this change).

**2. Stale NeuralWatt "unconfirmed access" comments corrected.** `scripts/evaluate_ragas.py`'s
`NEURALWATT_MODELS` module comment, `_neuralwatt_llm()`'s docstring and inline `NEURALWATT_BASE_URL`
comment, `run_neuralwatt_multi_judge_consensus()`'s "NOTE" docstring paragraph, and the
`--neuralwatt-judge-models` CLI flag's help text/print statements all previously described
NeuralWatt's base URL, OpenAI-API-compatibility, and model IDs as "UNCONFIRMED" / "not yet
confirmed" / "provisional" / "scaffolding only, NOT live-verified", and separately claimed
`run_neuralwatt_multi_judge_consensus()` had intentionally never been invoked against a live
NeuralWatt endpoint. This is now false: `/tmp/nfcorpus_eval_v2/neuralwatt_battery_v2_summary.txt`
documents a completed live 3-model x 6-file x 40-row battery (720 real judge calls) run via that
exact function, with a zero-NaN escalation chain (NeuralWatt -> Ollama fallback -> Claude
tiebreaker) resolving every row — `glm-5.2-short` 100% direct-NeuralWatt success, `kimi-k2.7-code`
99.2%, and `qwen3.5-397b` a confirmed real ~40% failure rate (29.6% ollama_fallback + 10.0%
claude_tiebreaker_escalation), not a hypothetical the escalation chain merely guards against. All
five call sites were updated to state the base URL (`https://api.neuralwatt.com/v1`), the three
model IDs, and NeuralWatt's OpenAI-API-compatibility as CONFIRMED and live-verified, citing the
battery artifact and `docs/CHANGELOG.md`. One narrower "not yet implemented" claim was
deliberately preserved rather than changed: `run_multi_judge_faithfulness_crosscheck()` (the
separate, Ollama-only cross-check function) still does not fold a NeuralWatt-backed judge into
its own model list — that remains a real, unbuilt extension point (`--neuralwatt-judge-models`),
now reworded to make clear it's a gap in *that specific function's* wiring, not in NeuralWatt
access itself (which `run_neuralwatt_multi_judge_consensus()` already exercises live). Verified via
`grep -i` across the whole file for "unconfirmed"/"not yet confirmed"/"provisional"/"scaffolding
only" language about NeuralWatt: no stragglers remain.

**Verification performed:**
- `python3 -m py_compile scripts/utils.py scripts/evaluate_ragas.py scripts/retrieve.py` — passes.
- Config-key parity check (AST-parse vs. `yaml.safe_load`, same method as every prior feature this
  session) confirms `rescore` is absent from both `config/config.yaml` and `utils.py`'s
  `default_config["retrieval"]` post-removal, with no new mismatches introduced (only the
  pre-existing `rrf_k` gap remains, unchanged).
- `grep -rn rescore` across the repo (excluding an unrelated stale worktree copy) returns no hits.
- `grep -in` for "unconfirmed"/"not yet confirmed"/"provisional"/"scaffolding only" in
  `scripts/evaluate_ragas.py` returns no hits about NeuralWatt access/confirmation status.

## 2026-07-19 — MMR (Maximal Marginal Relevance) diversity-aware selection pass added (`retrieval.mmr_enabled`, OFF by default, UNVALIDATED)

**What this targets:** `docs/ARCHITECTURE.md` §4.7 documented that the cross-encoder rerank stage
(`utils.py`'s `rerank_active` block in `EfficientRAG.retrieve()`) scores every candidate purely on
`(query, chunk_text)` relevance, with no term penalizing near-duplicate or heavily-overlapping
chunks — a gap flagged as plausible but lower-confidence/lower-priority than HyDE, specifically
because this project's NFCorpus-derived eval corpus was constructed as one gold document per query
unioned with randomly-sampled same-topic distractor documents (`MANIFEST.md`), a shape where
near-duplicate or overlapping content can plausibly crowd the top of a relevance-only ranking.
`docs/RESEARCH_NOTES.md` §4 independently re-verified the citations motivating this: **Carbonell &
Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing
Summaries," SIGIR '98** (https://dl.acm.org/doi/10.1145/290941.291025) — the foundational MMR
paper — and **Wang, Bi, Luo, Asur, Cheng, "Diversity Enhances an LLM's Performance in RAG and
Long-context Task," arXiv:2502.09017** (2025), which builds on MMR/FPS specifically for RAG content
selection and argues lightweight MMR is preferable to LLM-based diversity reranking on cost/latency
grounds. Both citations were confirmed correct (authors, venue, arXiv ID) with no inaccuracies
found.

**What was added, in `scripts/utils.py`:**

- `EfficientRAG._apply_mmr(candidates, query_embedding, lambda_param, top_n)` — standard iterative
  MMR: repeatedly picks the remaining candidate maximizing
  `lambda_param * relevance_to_query - (1 - lambda_param) * max_similarity_to_selected`, where both
  terms are cosine similarity over dense embeddings (`relevance_to_query` vs. the query embedding
  that drove the dense search leg; `max_similarity_to_selected` vs. whichever already-picked
  candidate is most similar, `0.0` for the first pick). A candidate missing its dense vector (e.g.
  a pre-existing edge case, not expected in normal operation) is treated as similarity `0.0` to
  everything rather than raising.
- **No new embedding round trip.** `_apply_mmr()` reuses each candidate's dense embedding, which is
  now carried through on a `"vector"` key added to the internal hit dicts inside `retrieve()`,
  itself populated from `qdrant.query_points(..., with_vectors=["dense"])` — the exact same
  dense-search calls `retrieve()` already makes for the hybrid/dense leg, just additionally
  requesting the stored vector back. This `with_vectors` request (and the `"vector"` key) is only
  populated when `retrieval.mmr_enabled` is true, so the default (off) path pulls exactly the same
  data over the wire as before this feature existed.
- Gate wiring in `EfficientRAG.retrieve()`: gated on `retrieval.mmr_enabled` (OFF by default).
  Positioned exactly where `docs/ARCHITECTURE.md` §1.2 documents the pipeline's step ordering as
  cross-encoder rerank -> parent-chunk expansion -> CRAG context review — MMR runs **immediately
  after** the rerank step's `top_hits` list is built (still child-chunk-scale, `rerank_top_n`
  candidates, default 8) and **before** parent-chunk expansion, so it never operates on
  already-expanded parent text (expanding first would make every candidate's text — and therefore
  any text-based diversity signal — far noisier; operating on dense embeddings sidesteps that, but
  the ordering still matters for candidate *scale*, per the task's own framing). The MMR pass
  reorders/selects from `top_hits` using `_apply_mmr(..., top_n=len(top_hits))` — i.e. v1 always
  keeps the full `rerank_top_n` candidate count, just reordered by the combined relevance+diversity
  score, rather than additionally truncating further (no new `mmr_top_n` config key was introduced;
  `rerank_top_n` already controls the pool size MMR operates on and reorders within). Each
  candidate's original relevance/fusion `score` field is preserved on the returned hit after MMR
  reordering — the MMR combination score is used only to pick the order, never surfaced as the
  hit's `score`, so `score`'s meaning is unchanged whether MMR is on or off.
- New config keys (`retrieval:` section, `config/config.yaml` AND `load_config()`'s
  `default_config` dict in `scripts/utils.py`, kept in sync per this project's existing
  dual-default pattern): `mmr_enabled` (`false`) and `mmr_lambda` (`0.6` — a relevance-leaning
  middle of the standard 0.5-0.7 RAG range cited in the literature above; chosen pending empirical
  tuning against this project's own NFCorpus harness, not derived from a specific ablation on this
  corpus).

**Judgment call on interaction with the Adaptive-RAG-lite `needs_broad_context` gate: kept
independent, not cross-wired, in v1.** The task prompt raised whether MMR should be suppressed for
queries the adaptive gate classifies as `needs_broad_context: true` (comparative/multi-part
queries that may genuinely need multiple distinct perspectives, where diversity-filtering could be
counterproductive). Decision: `mmr_enabled` is a plain, independent on/off toggle in v1, not gated
on `needs_broad_context`. Reasoning: (1) MMR operates on child-chunk candidates, `needs_broad_context`
gates parent-chunk *expansion* — they act on different representations at different pipeline stages,
so there's no structural conflict requiring immediate resolution; (2) this project's own established
convention (see the HyDE and query-adaptive-parent-expansion entries below) is to ship each new lever
as an independently toggleable flag first, and only wire a cross-feature interaction once there's
validated evidence it's needed — every prior cross-feature question in this project's history
(adaptive-gate + parent-expansion being the one exception, which had strong a priori reasoning tied
to a documented single-hop-vs-multi-hop research split) has otherwise been left decoupled pending
data; (3) neither MMR nor the adaptive gate has been A/B validated yet, so coupling them now would
mean tuning/debugging two unvalidated, interacting features at once instead of one at a time. If a
future validation run finds MMR measurably hurts genuinely-multi-perspective queries, the natural
follow-up is a new `mmr_skip_on_broad_context` (or similar) sub-toggle, mirroring
`adaptive_parent_expansion_enabled`'s pattern of a separate, independently-isolatable flag rather
than folding the interaction into `mmr_enabled` itself.

**Verification performed:**

- `python3 -m py_compile scripts/utils.py` — passes (via this project's `.venv`, since the bare
  system `python3` lacks `qdrant-client`/`sentence-transformers`/`fastembed`).
- Config-key parity check (AST-parse `load_config()`'s `default_config["retrieval"]` dict vs.
  `yaml.safe_load(config/config.yaml)["retrieval"]`, same method used for every prior feature this
  session): `mmr_enabled` and `mmr_lambda` present and matching on both sides. The one PRE-EXISTING
  `rrf_k` mismatch (§2 of `docs/ARCHITECTURE.md`) is unchanged — not newly introduced, not fixed.
- No-regression check, live: with `mmr_enabled: false` (the default), monkey-patched `_apply_mmr` to
  raise (`AssertionError`) if called, then ran `retrieve()` against the real
  `rag_nfcorpus_v2_adaptive_tuned` Qdrant collection for an NFCorpus-style query — completed
  successfully with 8 normal hits and `_apply_mmr` was never invoked, confirming the flag-off path is
  unaffected by this change.
- Synthetic unit test of `_apply_mmr()` in isolation (hand-built embeddings, no live calls): 5
  candidates — 3 near-duplicate ("A1/A2/A3", mutual cosine similarity ~0.995-0.9995, all clustered
  near the query direction) and 2 genuinely distinct ("B1/B2", different directions from the
  A-cluster and from each other), with pure-relevance order `A1 > A2 > A3 > B1 > B2`.
  - **`lambda_param=1.0`:** MMR order was `A1, A2, A3, B1, B2` — **identical to pure relevance
    order**, confirming the degenerate case (diversity term always multiplied by 0) behaves exactly
    as the algorithm predicts, not just "does something."
  - **`lambda_param=0.5`** (diversity-favoring): MMR order was `A1, A2, B1, A3, B2` — `B1` moved from
    relevance-rank 4 (0-indexed 3) up to MMR-rank 3 (0-indexed 2), confirming a genuinely distinct
    candidate was surfaced higher than pure relevance ranking would have placed it, once two
    near-duplicates (`A1`, `A2`) had already been selected and their redundancy penalized `A3`.
  - Edge cases (`candidates=[]`, `top_n=0`) both returned `[]` without error.
- Live end-to-end smoke test against the real `rag_nfcorpus_v2_adaptive_tuned` collection
  (default settings otherwise — hybrid search + rerank both on, `rerank_top_n=8`), `mmr_enabled: false`
  vs. `true`, `mmr_lambda=0.6` (the shipped default):
  - **"Does green tea consumption reduce the risk of gastric cancer?"** — same 6 unique source
    documents represented in both arms (top-8 is a permutation, not a set change, since
    `top_n=len(candidates)`), but MMR visibly separated a duplicate pair: OFF had two `MED-5256.md`
    chunks at ranks #1 and #8 and two `MED-4412.md` chunks at ranks #2 and #5; ON kept the same 8
    chunks but interleaved them differently, moving `MED-2584.md` (a distinct document) up from rank
    #6 to rank #2.
  - **"What are the effects of flaxseed on breast cancer survival?"** — a clearer, more concrete
    effect: OFF ranked the top TWO hits as two different chunks of the SAME document (`MED-4233.md`,
    ranks #1 and #2 — a genuine near-duplicate pair sitting immediately adjacent at the very top,
    exactly the failure mode §4.7/RESEARCH_NOTES.md §4 describes) before a third, different document
    (`MED-5341.md`) appeared at rank #3. With MMR ON, the second `MED-4233.md` chunk was pushed down
    to rank #4, and `MED-3866.md` — a genuinely different flaxseed-related document — moved up from
    rank #4 to rank #2, directly ahead of the redundant same-document chunk. This is a concrete,
    inspectable instance of MMR doing what it's designed to do: deprioritizing a same-source
    redundant chunk in favor of a distinct document's content.
  - In both queries the returned document **set** was unchanged (expected, since v1 always keeps
    `top_n=len(candidates)`, i.e. reorders rather than drops candidates) — the observed effect is
    entirely in **ordering**, which is what a downstream consumer that reads hits in rank order (or
    truncates to fewer than `rerank_top_n`) would actually see change.

**UNVALIDATED — read before assuming this improves end-to-end retrieval quality:** this is
implementation, unit-test, and smoke-test verification only, following the same scope as every
other UNVALIDATED entry in this file (HyDE, query-adaptive parent expansion, Adaptive-RAG-lite). The
actual A/B validation experiment (running the existing 40-question NFCorpus testset harness with
`mmr_enabled: true` vs. `false` and comparing RAGAS `context_precision`/`context_recall` — the axes
diversity-aware selection would plausibly affect, not necessarily faithfulness — via the paired
Wilcoxon + bootstrap CI methodology in `docs/ARCHITECTURE.md` §5) has explicitly **not** been run as
part of this change. Per this project's own stated priority order (`docs/ARCHITECTURE.md` §4.4's
decision record), MMR validation is lower priority than HyDE validation — this entry only
establishes that MMR is wired correctly, fails safely to a no-op when disabled, reuses existing
embeddings with no added embedding cost, is verified algorithmically correct via the lambda=1.0/0.5
unit test, and demonstrably changes candidate ordering (with a concrete same-document-redundancy
example) on real retrieval output when enabled.

## 2026-07-19 — Query-adaptive parent-chunk expansion added (`retrieval.adaptive_parent_expansion_enabled`, OFF unless the adaptive gate + `fetch_parents` are both on, UNVALIDATED)

**What this targets:** `docs/ARCHITECTURE.md` §4.6 / `docs/RESEARCH_NOTES.md` §6 documented that
`retrieval.fetch_parents` is a static global on/off toggle — every query gets the same
parent-chunk-expansion treatment regardless of whether it's a simple single-hop factoid question
(NFCorpus's dominant query style) or a genuinely multi-hop/comparative one. Chunking-granularity
research cited in `docs/RESEARCH_NOTES.md` §6 (this project's own synthesis, not one single
external paper — see that section for the citation-checking detail, including the corrected
attribution of the related ~44% RAGAS-faithfulness-invalid-rate figure to Kreileder/Reisinger/
Fischer, arXiv:2607.01852, not "Chroma Research") frames the general finding: expanding to a
larger parent chunk trades retrieval precision for broader context — helpful for multi-hop
questions that need to synthesize across sections, but actively diluting for single-hop factoid
questions where a precise ~100-token child match gets buried inside a noisier ~400-token parent
that includes less-relevant surrounding text. A static toggle cannot capture that split.

**What was added, in `scripts/utils.py`:** the Adaptive-RAG-lite gate's single classification
call (`_classify_retrieval_need()`) was extended to answer a SECOND, independent question in the
SAME LLM call — no redundant classifier round-trip added:

- New JSON field `needs_broad_context: bool`, alongside the existing `strategy: "none"|"light"`.
  True for queries that are comparative, multi-part, or require synthesizing across multiple
  facts/sections — the same criteria that were explicitly scoped OUT of `strategy` as a v1
  retrieval-*depth* "deep" tier (see the existing `_classify_retrieval_need()` docstring), but
  reused here for a DIFFERENT decision: whether to expand to parent-chunk text, not how many
  chunks to retrieve. Biased toward `false` when ambiguous (the conservative "keep the precise
  match" default), mirroring the gate's existing "bias toward light" lenient-default philosophy
  applied to this dimension's opposite direction.
- `_parse_retrieval_classification()` (renamed from `_parse_retrieval_strategy()`, single
  definition, no duplicate) parses both fields from one JSON response. `strategy` parsing is
  unchanged and still mandatory (raises `ValueError` if unparseable, letting the fail-open path
  take over); `needs_broad_context` is best-effort and defaults to `False` on any parse miss —
  a malformed/missing second field must never break the core "none"/"light" gate that already
  existed.
- `_classify_retrieval_need()`'s fail-open path (any call/parse/provider error) now also returns
  `needs_broad_context: False` alongside the existing fallback `strategy` — a classifier hiccup
  must degrade to "no query-adaptive expansion this time," never to force-expanding.
- `retrieve()` records the outcome on a new sibling instance attribute,
  `self._last_needs_broad_context` (`bool` | `None` if the gate didn't run), following the exact
  same inspectable-attribute pattern already established by `self._last_retrieval_strategy`.
- `retrieve()`'s existing parent-expansion block: when `retrieval.adaptive_retrieval_enabled` AND
  `retrieval.fetch_parents` are both true, the per-query `needs_broad_context` verdict (from the
  same classification call made earlier in the same `retrieve()` invocation — not a second LLM
  call) now decides whether the parent-chunk lookup actually runs for that query. If
  `adaptive_retrieval_enabled` is false, this sub-gate is entirely inert and `fetch_parents`
  behaves exactly as before — a static, non-query-adaptive toggle, byte-for-byte unchanged.
- New config key `retrieval.adaptive_parent_expansion_enabled` (default `true`), present in both
  `load_config()`'s `default_config` dict and `config/config.yaml` (parity-checked). This is
  deliberately a SEPARATE flag from `adaptive_retrieval_enabled`, not folded into it: a judgment
  call made so this specific sub-feature (query-adaptive parent expansion) can be isolated and
  toggled independently of the none/light retrieval-depth decision in a future validation
  experiment, without needing a code change to disentangle which half of the gate is responsible
  for a measured effect. Setting it `false` (with the gate still on) reverts to the old "always
  expand whenever `fetch_parents` is true" behavior, for that isolation use case.

**Verification performed:**
- `python3 -m py_compile scripts/utils.py` — passes.
- Config-key parity check (AST-parse `load_config()`'s `default_config` dict vs.
  `yaml.safe_load(config/config.yaml)`, same method used for the HyDE/adaptive-gate additions):
  `adaptive_parent_expansion_enabled` present in both. The pre-existing `retrieval.rrf_k` parity
  gap (§2 of `docs/ARCHITECTURE.md`) is unchanged by this work — not newly introduced, not fixed.
- No-regression check: with `adaptive_retrieval_enabled: false` (the default), the parent-expansion
  block reduces to `do_expand_parents = fetch_parents`, identical to the prior unconditional
  `if fetch_parents and formatted:` check — confirmed by direct code inspection of the new
  branch, not just by re-running a test.
- Live classifier smoke test (Ollama, `glm-5.2:cloud`, real network calls, no mocking): an
  unambiguous comparative query ("Compare the effects of flaxseed and green tea on cancer risk,
  and explain any differences in mechanism") consistently classified `needs_broad_context: true`
  across repeats. Unambiguous narrow single-fact queries ("What vitamin deficiency causes
  pellagra?", "What is the recommended daily intake of vitamin C?", "Does green tea consumption
  reduce the risk of gastric cancer?") consistently classified `needs_broad_context: false`.
  **Caveat found, not hidden:** a borderline query ("What are the effects of flaxseed on breast
  cancer survival?" — phrased close to "synthesize findings from multiple studies") flipped
  between `true` and `false` across repeated calls to the same cheap classifier model — real
  classifier noise on ambiguous phrasing, not a code defect. Clear-cut cases classify reliably;
  borderline ones don't, which any future A/B validation of this feature should account for
  (e.g. by not assuming per-query determinism run-to-run).
- Live end-to-end `retrieve()` smoke test against the real `nfcorpus_v2_adaptive_tuned` Qdrant
  collection (`config_adaptive_tuned.yaml` from `/tmp/nfcorpus_eval_v2/`, plus
  `adaptive_parent_expansion_enabled: true`): the single-fact query ("Does green tea consumption
  reduce the risk of gastric cancer?", classified `light`/`needs_broad_context: false`) returned
  hits with `text` and `child_text` at IDENTICAL lengths for all 4 inspected hits (185/185,
  115/115, 1443/1443, 1728/1728 characters) — confirmed NOT parent-expanded. The comparative query
  (classified `light`/`needs_broad_context: true`) returned hits where `text` was measurably
  LONGER than `child_text` for 3 of 4 hits (2598 vs 1387, 2598 vs 152, 1683 vs 1443 characters) —
  confirmed parent-expanded. This is a real text-length comparison from live retrieval output, not
  an assumption that the code path was "entered."

**IMPORTANT — implemented but statistically UNVALIDATED, and this is a fix for a known-stale
input, not a re-run of the thing it fixes:** this task does NOT re-run the Adaptive-RAG-lite A/B
validation experiment referenced in `docs/ARCHITECTURE.md` §4.1 (`adaptive_validation_summary.txt`
— the one whose own conclusion was *"Gating retrieval per-query DID NOT help close the
tuned-vs-baseline significance gap... The dilution hypothesis is NOT supported by this experiment
as run"*). That earlier experiment's `config_adaptive_tuned.yaml` used the OLD, since-superseded
character-derived chunk sizing later replaced by the 2026-07-16 token-based re-derivation
(`child_chunk_size: 400`/`parent_chunk_size: 1800` there, vs. the current token-based
`child_chunk_size: 100`/`parent_chunk_size: 400` defaults) — meaning that null result was measured
against a parent/child size ratio quite different from today's defaults, AND without any
query-adaptive parent-expansion capability at all (this capability didn't exist yet). This task
fixes the underlying capability gap (a query-adaptive expansion decision now exists); it does
**not** re-establish whether that dilution hypothesis holds under current, correctly-sized
chunking. **Follow-up task, not yet scheduled or executed:** re-run the Adaptive-RAG-lite A/B
experiment against the existing 40-question NFCorpus testset using CURRENT token-based chunk sizes
AND this new query-adaptive parent-expansion feature together (both `adaptive_retrieval_enabled`
and `adaptive_parent_expansion_enabled` on), using the same paired-Wilcoxon + bootstrap-CI +
3-repeat methodology as every other comparison in `docs/ARCHITECTURE.md` §4.1/§5, before citing any
faithfulness/precision delta for this specific combination.

## 2026-07-19 — HyDE (Hypothetical Document Embeddings) query rewriting added for the dense leg (`retrieval.hyde_enabled`, OFF by default, UNVALIDATED)

**What this targets:** `docs/ARCHITECTURE.md` §4.5 flagged HyDE (Gao et al., **arXiv:2212.10496**)
as this project's highest-expected-value untested technique, specifically because of a documented
vocabulary mismatch: NFCorpus queries are lay-phrased consumer-health questions
("What are the effects of flaxseed on breast cancer survival?") while the corpus documents are
formal clinical/scientific/PubMed-style text. Dense query-embedding similarity alone under-serves
that gap — the literal query and the document that should match it can use almost entirely
different vocabulary for the same concept — and hybrid search + cross-encoder reranking alone
can't bridge it either, since both still start from the literal query's embedding for the dense
leg.

**What was added, in `scripts/utils.py`:**

- `EfficientRAG._generate_hyde_document(query)` — a single cheap LLM call that writes a short
  (2-4 sentence) hypothetical passage that would answer `query`, "in the style of a factual
  reference document," then returns that passage's text. Provider dispatch reuses the exact same
  method as the Adaptive-RAG-lite gate's `_classify_retrieval_need()` — the shared dispatcher was
  renamed from `_call_adaptive_retrieval_llm` to the more accurate `_call_llm_provider` (call sites
  updated, no behavior change) rather than duplicating a second near-identical
  ollama/neuralwatt/claude dispatch block for HyDE to use. Same conventions as the adaptive gate:
  an unknown `retrieval.hyde_provider` raises `ValueError` immediately (a config typo is a loud
  failure), but any call/parse/network error on an otherwise-valid provider fails open, returning
  the **original query unchanged** — a HyDE hiccup degrades to "the dense leg embeds the literal
  query, exactly as if HyDE were off for this one call," never to failing the whole `retrieve()`.
- Gate wiring in `EfficientRAG.retrieve()`: gated on `retrieval.hyde_enabled` (OFF by default).
  When enabled, the DENSE leg's query text is replaced with `_generate_hyde_document(query)`'s
  output before embedding; the **sparse/BM25 leg and the cross-encoder rerank step are
  deliberately NOT touched** — both still use the literal `query` string. This is a design choice,
  not an oversight: HyDE is specifically a dense-embedding technique (Gao et al. embed the
  hypothetical document, not the query, precisely because dense similarity is what benefits from
  matching document-register-to-document-register); literal lexical/BM25 matching should still
  match the user's actual terms, and the final rerank step scores `(query, child_text)` pairs —
  i.e. the real retrieved passage against the real question — which is exactly what it should keep
  doing regardless of what was used to *find* the candidates.
- New config keys (`retrieval:` section, `config/config.yaml` AND `load_config()`'s
  `default_config` dict in `scripts/utils.py`, kept in sync per this project's existing
  dual-default pattern): `hyde_enabled` (`false`), `hyde_model` (`"glm-5.2:cloud"`, reusing
  `adaptive_retrieval_model`'s default), `hyde_provider` (`"ollama"`, reusing
  `adaptive_retrieval_provider`'s default).

**Deliberately kept as a separate LLM call from `_classify_retrieval_need()`, not merged:** both
features are independently toggleable (`hyde_enabled` / `adaptive_retrieval_enabled`, both off by
default) and serve different purposes — a routing decision vs. a query rewrite for embedding.
Merging them into one combined call would couple two features that each need to keep working
correctly when only ONE is enabled; a prompt/parsing change made for one could silently break the
other, and a combined call would pay for both prompts' worth of reasoning even when only one
feature is turned on for a given config. This reasoning is also captured directly in
`_generate_hyde_document()`'s docstring. Revisit combining them only if both are validated to help
independently AND the added latency of two sequential cheap calls is measured to matter — not
pre-emptively here.

**Verification performed (implementation + smoke-test only — see below for what was NOT run):**

- `python3 -m py_compile scripts/utils.py` — passes.
- Config-key parity check (`config/config.yaml`'s `retrieval:` keys vs. `load_config()`'s
  `default_config["retrieval"]` keys, same AST-parse + `yaml.safe_load` diff the adaptive-gate task
  used): all 3 new HyDE keys match exactly on both sides. The one PRE-EXISTING `rrf_k` mismatch
  (documented in the adaptive-gate entry below) is unchanged and remains the only gap.
- Gate-off regression check: with `hyde_enabled: false` (the default), monkey-patched
  `_generate_hyde_document` to raise if called, then ran `retrieve()` against the real
  `rag_nfcorpus_v2_adaptive_tuned` Qdrant collection (369 points, hybrid dense+sparse) for
  an NFCorpus-style query — confirmed the HyDE method is never invoked and `retrieve()` returns its
  normal 8 hits exactly as before this change.
- Live smoke test of `_generate_hyde_document()` alone (provider `"ollama"`, model
  `glm-5.2:cloud`, no `NEURALWATT_API_KEY`/Anthropic credentials needed) against 3 real
  NFCorpus-style medical questions — generated passages read as plausible, on-register factual
  reference text, not garbage or refusals:
  - "What are the effects of flaxseed on breast cancer survival?" → "Dietary flaxseed, which is
    rich in lignans and omega-3 fatty acids, has been investigated for its potential role in
    modulating breast cancer progression and patient survival. Observational studies suggest that
    higher dietary lignan intake is associated with a modest reduction in all-cause and breast
    cancer-specific mortality among postmenopausal breast cancer survivors. ..."
  - "Can drinking green tea reduce the risk of stomach cancer?" → "Epidemiological studies
    investigating the relationship between green tea consumption and stomach cancer risk have
    yielded inconsistent results. While in vitro and animal models suggest that green tea
    polyphenols, particularly epigallocatechin gallate (EGCG), exhibit chemopreventive
    properties, ..."
  - "Does vitamin D deficiency cause depression?" → "While epidemiological studies consistently
    demonstrate an association between low serum 25-hydroxyvitamin D concentrations and an
    increased prevalence of depressive symptoms, a definitive causal relationship remains
    unproven. ..."
- End-to-end smoke test through `retrieve()` against the same live 369-point collection for the
  flaxseed query, `hyde_enabled: true` vs. `false`, default settings (hybrid search + rerank both
  on): both paths completed without error and returned 8 non-empty hits each; the final reranked
  top-8 was identical between the two runs on this particular query/corpus (expected/fine — the
  cross-encoder rerank step re-scores the same fused dense+sparse candidate pool against the
  literal query, so a small corpus can converge on the same final ranking even when the dense
  leg's candidate *retrieval* differs). To directly confirm the dense-leg substitution itself is
  live (not a silent no-op), re-ran with `rerank: false` and `hybrid_search: false` (isolating the
  raw dense-only ranking): cosine similarity scores were uniformly higher with HyDE enabled (e.g.
  top score 0.845 → 0.8922) and the 4th-ranked result changed from `MED-3866` to `MED-5111` —
  direct evidence the dense leg is genuinely embedding the generated hypothetical document instead
  of the literal query when the flag is on.

**UNVALIDATED — read before assuming this improves end-to-end retrieval quality:** this is
implementation and unit/smoke-testing only, following the exact same "implement, verify wiring,
smoke-test on hand-picked examples" scope as the Adaptive-RAG-lite entry below. The actual A/B
validation experiment (running the existing 40-question NFCorpus testset harness with
`hyde_enabled: true` vs. `false` and comparing RAGAS metrics via the paired Wilcoxon + bootstrap CI
methodology in `docs/ARCHITECTURE.md` §5) has explicitly NOT been run as part of this change — that
is a separate follow-up task. Do not treat this entry as evidence HyDE improves
`context_precision`/`context_recall`/downstream faithfulness on this corpus; it only establishes
that the technique is wired correctly, fails open safely, defaults to a no-op, generates
plausible hypothetical documents, and demonstrably changes the dense leg's candidate ranking when
enabled.

## 2026-07-19 — `docs/RESEARCH_NOTES.md` added: citations for the six candidate-fix research questions re-verified

A prior research investigation (six questions, run via parallel web research ahead of the
HyDE/MMR/statistical-power decisions referenced elsewhere in `docs/ARCHITECTURE.md` §4 and in this
file's other 2026-07-19 entries) had never been saved anywhere in this repo — its conclusions were
already baked into `docs/ARCHITECTURE.md` §4.3-§4.8 and this file's own prior entries (e.g. the
Self-Correcting RAG citation, arXiv:2604.10734, in the 2026-07-16 verify-and-repair and
context-review entries below), but with no way for a future reader to check the underlying
citations without re-running the research from scratch. A documentation-review pass flagged this
as a real gap: claims citing specific papers/arXiv IDs (RAGAS faithfulness mismatch, HyDE, MMR,
RouteLLM/MoA/Self-MoA, a chunking-ablation study) were being repeated without a checkable trail.

**What was added:** `docs/RESEARCH_NOTES.md` — each of the six original research questions
re-verified independently via web search (not by trusting the original summary), with full
citations (arXiv IDs, authors, venues) and an explicit verified/unverified/inaccurate verdict per
finding. **One citation was found to be actively wrong and is flagged in that document:** the
"~44% invalid RAGAS faithfulness score rate in chunking-ablation contexts" figure was attributed
to "Chroma Research" in the original summary, but is actually from Kreileder, Reisinger & Fischer,
arXiv:2607.01852 — Chroma's own actual published chunking-evaluation technical report
(trychroma.com/research/evaluating-chunking) was checked directly and does not use RAGAS or
faithfulness scoring at all. Two other findings (NFCorpus dense-vs-hybrid recall numbers, an
LLM-reranking ablation gain) had their specific numbers confirmed accurate but traced to
non-peer-reviewed sources (an unreviewed single-author ResearchGate preprint, and a
student-authored paper in a low-prestige journal) — flagged as suggestive rather than
authoritative. The remaining findings (RAGAS/DeepEval faithfulness mechanism, HyDE, MMR,
RouteLLM/MoA/Self-MoA) all had their arXiv IDs and core claims fully confirmed against primary
sources. See `docs/RESEARCH_NOTES.md` for the full per-finding writeup and verification detail.

## 2026-07-19 — DECISION: N=40 statistical-power gap — accept the ceiling, redirect to HyDE (not testset expansion or corpus switch)

**The decision this closes:** `docs/ARCHITECTURE.md` §4.4 tracked, as an open decision task, what
to do about the finding that the ~40-question NFCorpus harness (`/tmp/nfcorpus_eval_v2/`) is
roughly 30-100x underpowered to reliably detect the ~0.02 mean-faithfulness deltas actually being
measured for hybrid+rerank tuning (formal power analysis: ~N=1,200 needed for that effect size,
given the observed per-question judge-noise variance), while published NFCorpus/BEIR-family
literature independently reports near-zero real hybrid+rerank gains on this exact benchmark
family — meaning the repeated null results already logged in §4.1 may be the textbook-correct
answer for this technique on this benchmark, not evidence of a broken or mistuned pipeline.

**Options evaluated:**

- **(a) Expand the testset** toward the informal ~200-400 RAG-eval norm, or the full ~1,200 needed
  for the observed effect size. Checked `/tmp/nfcorpus_eval_v2/expand_testset.py` (the script
  already used for the 20→40 expansion) and `MANIFEST.md`: the method generalizes cleanly to more
  queries (new seed, skip already-used question texts, draw from the same
  `BeIR/nfcorpus-qrels["test"]` pool), and that pool has **~323 valid candidate query IDs total**
  (per the script's own printed diagnostic during the 20→40 run), of which only 40 are consumed —
  so ~300-320 total questions is mechanically reachable without changing methodology, but the full
  ~1,200 target is NOT (it exceeds the entire valid-qrels pool by ~4x and would require a different,
  unvalidated construction method — e.g. admitting lower-relevance-score judged docs as gold).
  Even the reachable ~300-320 ceiling would cost multiple days of compute (NeuralWatt 3-model panel
  wall-clock at N=40 was already ~60-100 min/config-repeat, and the methodology requires 3 repeats
  x multiple arms x multiple judges) to more precisely measure an effect independent literature
  already suggests is close to zero.
- **(b) Accept the ceiling; redirect to HyDE.** Required N scales with `(noise/effect)²`, so a
  technique with an order-of-magnitude-larger expected effect (HyDE: cited +40-58% gains on
  comparable corpora, vs. hybrid+rerank's observed ~0.02) needs roughly two orders of magnitude
  fewer questions for equivalent power — the existing N=40 testset is plausibly *already* adequate
  to detect a real HyDE effect, at zero additional testset cost. This also matches this project's
  pre-existing priority ordering (`docs/ARCHITECTURE.md` §4.5 already flagged HyDE as the
  highest-expected-value untested technique).
- **(c) Switch benchmark corpus.** Rejected: the literature finding motivating this whole decision
  is a BEIR-**family** pattern, not NFCorpus-specific, so there's no basis to expect a different
  BEIR corpus would surface a bigger hybrid+rerank effect — switching would likely reproduce the
  same null on a different dataset while losing comparability with every existing §4.1 result and
  requiring a brand-new, unvalidated gold-document/reference-proxy construction method.

**Decision: (b) is adopted as the primary path — stop treating null hybrid+rerank/faithfulness
results at N=40 as diagnostic of pipeline quality, and prioritize implementing/validating HyDE next
against the existing `testset_v2_40q.json` using the established 3-repeat/paired-Wilcoxon/bootstrap
methodology (§5).** A bounded, explicitly lower-priority and not-yet-scheduled version of (a) is
also recorded as a future option: if a future need arises for a tighter CI (e.g. to re-test
hybrid+rerank once more or to corroborate a HyDE result), extend `expand_testset.py` with a new
seed (e.g. `SEED=123`) to grow toward ~120-150 questions (~3-4x, mechanically identical to the
20→40 expansion, writing a new `testset_v2_120q.json` without touching `testset_v2_40q.json`) —
this narrows the CI by ~0.58x, enough to rule out moderate (~0.05+) effects, at roughly linear
(not 30x) cost. (c) is closed, not open — do not re-propose a corpus switch as a fix for §4.1's
null results without new evidence contradicting the BEIR-family literature finding above.

**No code or testset changes were made in this decision task** — this is a documentation-only
decision recorded in `docs/ARCHITECTURE.md` §4.4 (which this changelog entry summarizes) and here.
The concrete next step (HyDE implementation/validation) is a separate, not-yet-started follow-up
task.

## 2026-07-19 — Adaptive-RAG-lite: query-adaptive retrieval gate added (none/light only, v1 scope, UNVALIDATED)

**Root-cause hypothesis this targets (the "dilution hypothesis"):** across multiple statistical
rigor passes and multiple judges/repeats in this project's evaluation work (RAGAS metrics, the
multi-judge NeuralWatt/Ollama faithfulness cross-checks, the human review tier), tuned-vs-baseline
retrieval configurations keep coming out statistically indistinguishable from each other. One
plausible explanation that had not yet been tested: the tuning (reranking, hybrid search,
parent-chunk expansion, context review, etc.) applies UNIFORMLY to every query, including queries
that don't need deep retrieval at all — or don't need retrieval at all, being answerable from
general knowledge. If a meaningful fraction of any real testset's queries fall into that bucket,
uniformly-applied retrieval tuning has nothing to bite on for those queries, which could dilute an
aggregate metric delta down to noise even if the tuning genuinely helps on the queries that do need
it. Adaptive-RAG-lite is a first, deliberately minimal test of that hypothesis: route each query
through a single cheap classification call before the retrieval pipeline runs, rather than treating
every query identically.

**What was added:**

- `EfficientRAG._classify_retrieval_need(query)` (`scripts/utils.py`) — a single cheap LLM call
  classifying a query as `"none"` (answerable from general/common knowledge, no retrieval needed)
  or `"light"` (normal retrieval, current default settings), returning
  `{"strategy": "none"|"light", "raw_response": str}`. Biased toward `"light"` when ambiguous
  (same lenient-default philosophy as `context_review_threshold=0.3` elsewhere in this file):
  wrongly skipping retrieval is worse than wrongly doing an unnecessary one. Provider dispatch
  (`retrieval.adaptive_retrieval_provider`: `"ollama"` | `"neuralwatt"` | `"claude"`) mirrors the
  existing lazy cross-module import pattern `review_and_distill_context()` already uses to reach
  `evaluate_ragas.py`'s `_build_anthropic_client()` / `_neuralwatt_llm()` (and, for `"ollama"`,
  the same `OLLAMA_BASE_URL` env-var pattern as `_ollama_judge_llm()`) without a circular import.
  An unknown provider value raises `ValueError` immediately — a config typo is a loud failure, not
  something folded into the fail-open path. Any call/parse/network error on an otherwise-valid
  provider DOES fail open, returning `retrieval.adaptive_retrieval_fallback_strategy` (default
  `"light"`) — a classifier hiccup must degrade to "the gate did nothing this time," never to
  silently skipping retrieval. Response parsing (`_parse_retrieval_strategy()`) tolerates strict
  JSON, a markdown-fenced JSON block, leading/trailing whitespace, and a bare unquoted
  `none`/`light` value before giving up and falling back.
- Gate wiring in `EfficientRAG.retrieve()`: gated on `retrieval.adaptive_retrieval_enabled` (OFF
  by default). When enabled, runs the classifier BEFORE the hybrid search / rerank / parent
  expansion / context review pipeline. `"none"` short-circuits to `return []` immediately,
  skipping the whole pipeline; `"light"` overrides `top_k`/`oversampling` for that one call via
  `adaptive_light_top_k`/`adaptive_light_oversampling` (a local shallow-copied config dict, so
  `self.config` is never mutated) and then proceeds through the normal pipeline unchanged. The
  outcome is recorded on a new instance attribute, `self._last_retrieval_strategy` (`"none"` |
  `"light"` | `None` when the gate is off), rather than changing `retrieve()`'s return
  type/shape — a non-breaking addition for every existing caller.
- New config keys (`retrieval:` section, `config/config.yaml` AND `load_config()`'s
  `default_config` dict in `scripts/utils.py`, kept in sync per this project's existing
  dual-default pattern): `adaptive_retrieval_enabled` (false), `adaptive_retrieval_model`
  (`"glm-5.2:cloud"`), `adaptive_retrieval_provider` (`"ollama"`), `adaptive_light_top_k` (8),
  `adaptive_light_oversampling` (3.0), `adaptive_retrieval_fallback_strategy` (`"light"`).
- `scripts/retrieve.py`'s `retrieve_context()` reads `rag._last_retrieval_strategy` after calling
  `rag.retrieve(...)` and adds two new fields to its returned dict: `retrieval_skipped: bool` and
  `retrieval_strategy: Optional[str]`. `format_context()` gained a `retrieval_skipped` parameter
  so a deliberately-skipped-by-the-gate query produces a message clearly distinguishable from the
  pre-existing "No relevant context found in knowledge base." (which means retrieval ran and came
  up empty) — the new message reads "No retrieval performed — query classified as answerable from
  general knowledge (Adaptive-RAG-lite gate, retrieval.adaptive_retrieval_enabled)."

**v1 scope, deliberately trimmed:** binary `"none"` vs `"light"` only — no third `"deep"` tier
(deferred to a future pass; the naming convention, e.g. `adaptive_light_top_k`, is kept consistent
with a `deep` tier if one is added later, but `adaptive_deep_top_k`/`adaptive_deep_oversampling`
are intentionally NOT added yet). Self-RAG and FLARE were both considered and ruled out as
infeasible with this project's current infrastructure: Self-RAG requires a fine-tuning pipeline
this project doesn't have, and FLARE requires streaming generation with token-level confidence
signals this project's generation path doesn't support either. Adaptive-RAG-lite's single
before-the-fact classification call is the only one of the three actually implementable without
new infrastructure.

**Verification performed (implementation + smoke-test only — see below for what was NOT run):**

- `python3 -m py_compile scripts/utils.py scripts/retrieve.py` — passes.
- Config-key parity check (`config/config.yaml`'s `retrieval:` keys vs. `load_config()`'s
  `default_config["retrieval"]` keys): all 6 new adaptive keys match exactly on both sides. (One
  PRE-EXISTING, unrelated mismatch was found and left as-is: `rrf_k` is present in
  `config/config.yaml` but not in `load_config()`'s `default_config` dict — `EfficientRAG.__init__`
  already has its own runtime default of 60 for this key, so it's not a regression risk, just a
  pre-existing gap outside this task's scope.)
- Gate-off regression check: with `adaptive_retrieval_enabled: false` (the default), monkey-patched
  `_classify_retrieval_need` to raise if called, then ran `retrieve()` against a real Qdrant
  collection — confirmed the gate method is never invoked, `_last_retrieval_strategy` stays `None`,
  and results come back exactly as before this change.
- Live smoke test of `_classify_retrieval_need()` alone (provider `"ollama"`, model
  `glm-5.2:cloud`, no `NEURALWATT_API_KEY`/Anthropic credentials needed) against 3 hand-picked
  queries — classifications look sane:
  - "What is the capital of France?" → `"none"` (`reasoning`: "The capital of France is a basic,
    widely known fact that does not require retrieval from a specific corpus.")
  - "How many legs does a spider have?" → `"none"` (`reasoning`: "The number of legs a spider has
    is a basic, well-known biological fact that does not require retrieving information from a
    specific corpus.")
  - "What are the effects of flaxseed on breast cancer survival?" (an NFCorpus-style
    medical/nutrition query) → `"light"` (`reasoning`: "The query asks about the specific effects
    of flaxseed on breast cancer survival, a specialized medical topic that likely depends on
    specific clinical studies or ingested health data.")
- End-to-end smoke test through `retrieve_context()` with the gate enabled against a throwaway
  ingested collection: the `"none"` query returned `retrieval_skipped=True`,
  `retrieval_strategy="none"`, `num_results=0`, and the new distinct formatted message; the
  `"light"` query returned `retrieval_skipped=False`, `retrieval_strategy="light"`, and normal
  non-empty results — matching the documented contract.

**UNVALIDATED — read before assuming this fixes the dilution problem:** this is implementation and
unit/smoke-testing only. The actual A/B validation experiment (running the existing NFCorpus
testset harness with `adaptive_retrieval_enabled: true` vs. `false` and comparing RAGAS
metrics/statistical significance) has explicitly NOT been run as part of this change — that is a
separate follow-up task. Do not treat this entry as evidence the dilution hypothesis is correct or
that Adaptive-RAG-lite improves end-to-end quality; it only establishes that the gate is wired
correctly, fails open safely, defaults to a no-op, and produces sane classifications on a small
hand-picked set of obviously-easy cases.

## 2026-07-17 — NeuralWatt multi-judge consensus: zero-NaN guarantee via per-row escalation (hard requirement, live-verified)

**The gap this closes:** the entry directly below this one added an Ollama-cloud fallback for
`_score_one_model()`, but that fallback only triggers on an **exception** raised from the WHOLE
`evaluate()` call for a judge model. ragas has its own internal per-row retry logic; when it
exhausts those retries for one specific row, it does **not** raise anything — it just leaves `NaN`
in that row's faithfulness score and returns normally, same as any other row. This means the
existing fallback could never catch these silent per-row NaNs, since no exception was ever raised
for it to catch. **Confirmed live** (see the entry below): a 5-row NeuralWatt run had 2/5 rows NaN
for `qwen3.5-397b` even with the whole-dataset fallback active.

**Requirement (hard bar, not "fewer NaN"):** the final aggregated per-question output must never
contain a NaN for any judge that was supposed to produce a score.

**Fix, in `run_neuralwatt_multi_judge_consensus()`** (`scripts/evaluate_ragas.py`): after each
judge's `evaluate()` call completes and per-row faithfulness scores are extracted, every row that
is still `None` (NaN) is retried individually — never the whole batch — through an escalation
chain, in priority order:

1. **Single-row Ollama-cloud retry.** Only attempted when the whole dataset ran on NeuralWatt
   primary for that model (i.e. this specific row is a genuine ragas-internal silent NaN, not
   already an Ollama result from the whole-dataset-level fallback). Built as a fresh single-row
   `EvaluationDataset` containing just that one `SingleTurnSample`, evaluated on its own with a
   fresh `Faithfulness()` instance — mirrors `_claude_tiebreaker_faithfulness()`'s existing
   one-sample pattern rather than inventing a new one. Uses the same
   `_NEURALWATT_TO_OLLAMA_FALLBACK` mapping already in place. If the whole dataset had *already*
   fallen back to Ollama for that model (this row was NaN there too), or the whole model failed
   outright, this step is skipped — retrying the identical model+row combination again would not
   change the outcome.
2. **Claude tiebreaker as the final backstop** (`_claude_tiebreaker_faithfulness()`, already
   implemented for the separate high-disagreement/all-judges-failed triggers, now also invoked
   for these NaN-escalation cases).
3. **Loud, distinct "unrecoverable" flag** if even the Claude tiebreaker fails (e.g. no
   `ANTHROPIC_API_KEY`/Vertex access) — the row is logged immediately with a `print()` naming the
   model, row index, and truncated question text, the score stays `None`, and the per-question
   dict carries `"score_unrecoverable": True`. This is an acceptable final failure mode, but a
   **visible** one — never a silently reintroduced NaN.

**Per-question output changes:** each `per_question` row dict gained a `"judge_used"` field
(`Dict[model_name, str]`) recording the full escalation-chain outcome per judge — `"neuralwatt"` /
`"ollama_fallback"` / `"claude_tiebreaker_escalation"` / `"unrecoverable"` — and a
`"score_unrecoverable"` bool (`True` if any judge's chain for that row ended unrecoverable). The
existing `geometric_median`/`delta_m`/`high_disagreement`/`suspiciously_unanimous`/
`all_judges_failed`/`claude_tiebreaker` fields are unchanged in shape but now compute from the
POST-escalation scores, so `all_judges_failed` should now only ever be `True` in the extreme edge
case where every judge's entire chain (NeuralWatt → Ollama fallback → Claude tiebreaker) failed for
that row. The pre-existing high-disagreement/all-judges-failed-triggered Claude tiebreaker call
(a separate consensus mechanism, distinct from the per-row NaN-escalation tiebreaker calls) is
unchanged and still fires independently on top of this.

**Verified live (hard zero-NaN bar, not just "improved"):** ran the fixed
`run_neuralwatt_multi_judge_consensus()` against the same 5-row slice that showed the NaN problem
(`/tmp/nfcorpus_eval_v2/results2_tuned_r1_5row.csv`, first 5 rows of `results2_tuned_r1.csv`) with
`NEURALWATT_API_KEY` set for this run only. Result: **0/15 NaN** (3 models × 5 rows) — confirmed
by inspecting every row's `neuralwatt_scores` dict directly (all three judges' scores present and
non-`None` on every row). One row (`qwen3.5-397b` on the "Flaxseeds & Breast Cancer Survival"
question) hit a `TimeoutError` during the primary NeuralWatt call and correctly escalated straight
to `claude_tiebreaker_escalation` (`judge_used["qwen3.5-397b"] == "claude_tiebreaker_escalation"`,
final score `0.846`) rather than being left as NaN or silently dropped. Every other
(model, row) pair scored directly via `"neuralwatt"`. `score_unrecoverable` was `False` on all 5
rows. Total wall-clock time for the 5-row run: 934.8s (~15.6 min) — longer than the pre-fix 5-row
run (~11.9 min) because of the added single-row escalation call for the timed-out row, an accepted
cost of the zero-NaN guarantee. `python3 -m py_compile scripts/evaluate_ragas.py` passes.

**This is now a hard guarantee, not an improvement:** the only way a row's score can still be
`None` after this fix is if NeuralWatt, its Ollama-cloud fallback, AND the Claude tiebreaker all
failed for that specific row — and even then, `score_unrecoverable: True` makes that failure loud
and visible rather than a silent NaN blending into aggregate statistics.

## 2026-07-17 — NeuralWatt multi-judge consensus: shared-singleton race, uncapped concurrency, aggregation logic bugs fixed; live-verified; Ollama fallback added

Two independent passes (a debug/root-cause investigation and a structured code review) against
`run_neuralwatt_multi_judge_consensus()` and its helpers, confirmed against the real installed
`ragas==0.3.9` source and a live overnight run's monkey-patch workaround (`/tmp/nfcorpus_eval_v2/
run_one_neuralwatt.py`, never checked into this repo), surfaced 4 must-fix bugs, 3 should-fix
issues, and 2 nits. All are fixed below, plus one further fix (an Ollama-cloud fallback) added
mid-verification in response to live instability this same work uncovered.

### M1 — Shared mutable singleton race (Must-fix, root cause of the overnight run's 100% NaN failures)

**Symptom (from the overnight driver script's own inline comment, `run_one_neuralwatt.py`):** a
full run of `results2_baseline_r1.csv` at `concurrency=3` (the function's own default) returned
100% NaN faithfulness for all 3 NeuralWatt models — every one of 120 judge calls failed with
either `AssertionError("llm must be set to compute score")` or spurious `APIConnectionError`.

**Root cause:** `scripts/evaluate_ragas.py` imports a single module-level `faithfulness` object
once at import time (`from ragas.metrics import ... faithfulness ...`). `_score_one_model()`
(inside `run_neuralwatt_multi_judge_consensus()`), `_claude_tiebreaker_faithfulness()`, and
`run_multi_judge_faithfulness_crosscheck()`'s per-model loop all passed this SAME shared object
into `evaluate(metrics=[faithfulness], ...)`. Reading ragas 0.3.9's actual source confirmed
`evaluate()` mutates `metric.llm = llm` in place as part of run setup — a check-then-act race
when 3 threads do this concurrently on the identical object with 3 different judge LLMs. The
overnight driver worked around this at the call site (without touching this file, since it was
told not to) by serializing the outer panel to `concurrency=1`, sacrificing all 3-way
parallelism to sidestep the race rather than fixing it.

**Fix:** `_score_one_model()`, `_claude_tiebreaker_faithfulness()`, and
`run_multi_judge_faithfulness_crosscheck()`'s loop now each construct a **fresh `Faithfulness()`
instance** per call (`from ragas.metrics import Faithfulness`, confirmed importable/instantiable
against the installed `ragas==0.3.9`) instead of reusing the shared singleton.
`run_multi_judge_faithfulness_crosscheck()` runs its per-model loop sequentially today (no
actual race yet) but shares the identical bug pattern and its own docstring already documents
parallelization as a plausible future extension — fixed proactively so that extension doesn't
silently reintroduce this race. Its concurrency model (sequential, no `ThreadPoolExecutor`) was
NOT otherwise changed.

### M2 + concurrency design (Must-fix + resolves the review's S4 finding) — uncapped internal concurrency, now confirmed-safe

**Symptom:** neither `evaluate()` call site in this file passed `run_config=`, so each silently
got ragas's own `RunConfig(max_workers=16)` default.

**The review's open question:** naively using `RunConfig(max_workers=3)` inside each of 3
concurrently-running judge threads (outer `ThreadPoolExecutor(max_workers=3)`) could produce up
to 3×3=9 concurrent NeuralWatt requests, not 3 — and it was UNKNOWN whether NeuralWatt's rate
limit was per-model or account-wide.

**Now confirmed (user-verified directly, not an assumption):** NeuralWatt's rate limit is **3
concurrent requests ACCOUNT-WIDE**, not per-model.

**Design, confirmed correct by the above:** `RunConfig(max_workers=1)` INSIDE each judge thread
(`_NEURALWATT_JUDGE_MAX_WORKERS = 1`, a module-level constant with its reasoning documented at
the definition site) combined with the OUTER `ThreadPoolExecutor(max_workers=concurrency)`
(default `concurrency=3`, unchanged, genuine 3-way panel parallelism preserved). This allocates
exactly one of NeuralWatt's 3 account-wide slots to each judge model — total concurrent
NeuralWatt requests never exceeds 3, matching the confirmed limit exactly. The outer panel was
deliberately NOT serialized to `concurrency=1` (that was the overnight driver's overcorrection,
explicitly called out as wrong per this task's instructions).
`run_multi_judge_faithfulness_crosscheck()`'s Ollama loop also now passes an explicit
`run_config=RunConfig(max_workers=_OLLAMA_JUDGE_MAX_WORKERS)` (=16, ragas's own historical
default, since Ollama is local with no observed concurrency ceiling) — added purely for
consistency/auditability across all three touched `evaluate()` call sites, not because Ollama
needed a lower cap.

### M3 — Tiebreaker given its own concurrency constant (Must-fix)

`_claude_tiebreaker_faithfulness()` always scores exactly one sample, so its own
`RunConfig(max_workers=_CLAUDE_TIEBREAKER_MAX_WORKERS)` (=1) is a no-op today either way — but
it's a SEPARATE named constant from `_NEURALWATT_JUDGE_MAX_WORKERS`, specifically so a future
change that parallelizes/batches the (currently sequential) per-question tiebreaker loop doesn't
silently inherit a value tuned for NeuralWatt's confirmed rate limit instead of Claude's.

### M4/S2 — Config errors vs. transient failures conflated (Must-fix/Should-fix)

**Symptom:** `_neuralwatt_llm()` raises `RuntimeError` with a loud, actionable message when
`NEURALWATT_API_KEY` is unset (by design). `_score_one_model()`'s broad
`except Exception as exc: # noqa: BLE001` caught this identically to a transient
network/API failure, demoting it to a quiet `{"error": str(exc)}` — if the key is simply unset,
all 3 models fail identically with zero signal that it's a one-line env-var fix vs. 3
independent outages.

**Fix:** the error dict now carries `"error_type": "config" if isinstance(exc, RuntimeError)
else "runtime"`. The broad `except Exception` itself was kept (intentional, matching
`run_multi_judge_faithfulness_crosscheck()`'s sibling "one bad judge shouldn't kill the others"
design) — only the resulting dict now distinguishes the two cases.

### S3 — No timeout/retry config on the NeuralWatt client (Should-fix)

`_neuralwatt_llm()`'s `ChatOpenAI(...)` construction set neither a request timeout nor a retry
count. Added `request_timeout=90, max_retries=2` as a mitigation for observed Cloudflare 524
gateway timeouts on at least one NeuralWatt model — explicitly NOT a full fix for genuine
backend instability (see the live-verification findings below, where this mitigation's
interaction with a persistently slow model was directly observed).

### S5 — `suspiciously_unanimous` unconditionally True with only 1 of 3 judges succeeding (Should-fix, real logic bug)

**Bug:** with exactly 1 valid score, `delta_m` was hardcoded to `0.0`, which made
`suspiciously_unanimous` evaluate `True` unconditionally (any positive `unanimous_tolerance`
exceeds 0.0) even though a single judge cannot "agree" with itself. Post-fix, this was expected
to occur often, since `qwen3.5-397b` has a genuinely high real-world failure rate independent of
these bugs — confirmed true in the live smoke test below (see "Live verification").

**Fix:** `suspiciously_unanimous` and `high_disagreement` are now only computed from a real
spread when `len(valid_scores) >= 2`; with 0 or 1 valid scores, both flags are `False` and
`delta_m` is `None` (not `0.0`), making "insufficient data to assess agreement" distinguishable
from "these judges genuinely agreed" (a real `delta_m` of `0.0` from 2+ judges).

### S6 — All-judges-failed rows never got a Claude tiebreaker (Should-fix, real logic gap)

**Bug:** when `len(valid_scores) == 0`, `delta_m` was `None`, and `high_disagreement = delta_m
is not None and delta_m > threshold` short-circuited to `False` — the row where the ENTIRE
primary panel failed (most in need of a tiebreaker) was indistinguishable from a boring
low-disagreement row and never got one.

**Fix:** added `all_judges_failed = len(valid_scores) == 0` and
`needs_tiebreaker = high_disagreement or all_judges_failed`; the tiebreaker now fires on either
condition, and `all_judges_failed` is a new field in each `per_question` row dict (also
surfaced in the CLI's `--neuralwatt-multi-judge-crosscheck` output as an "All-judges-failed
rows: N / M" count line and folded into the high-disagreement detail listing).

### N1 — Docstring updated for corrected agreement-flag semantics (Nit)

`run_neuralwatt_multi_judge_consensus()`'s docstring previously claimed `high_disagreement`/
`suspiciously_unanimous` are always mutually exclusive "given disagreement_threshold >
unanimous_tolerance" — true only when `len(valid_scores) >= 2` post-S5-fix (with 0 or 1 valid
scores both flags are always `False`). Docstring and `Returns` section updated accordingly; the
tiebreaker-trigger condition (point 6) also updated to mention `all_judges_failed`.

### N4 — no further action needed

M1's fresh-`Faithfulness()`-per-call fix was backported into
`run_multi_judge_faithfulness_crosscheck()` as part of M1 above; nothing further was needed
there.

### Verification performed

1. `python3 -m py_compile scripts/evaluate_ragas.py` — passes.
2. Isolated unit-level check of the S5/S6 aggregation logic (the exact block extracted and run
   standalone, no live `evaluate()` calls): `[0.8]` (1 valid) → `suspiciously_unanimous=False`,
   `delta_m=None` (NOT the old buggy `True`/`0.0`) — **PASS**. `[]` (0 valid) →
   `all_judges_failed=True`, tiebreaker-trigger condition `True` — **PASS**. `[0.7, 0.7, 0.7]`
   (3 valid, real agreement) → `suspiciously_unanimous=True`, `delta_m=0.0` — **PASS**.
   `[0.1, 0.9, 0.5]` (3 valid, real disagreement) → `high_disagreement=True`,
   `suspiciously_unanimous=False` — **PASS**. All 4 cases passed.
3. **Live smoke test** against the real 40-question testset
   (`/tmp/nfcorpus_eval_v2/results2_tuned_r1.csv`, the same file the overnight run scored) with
   `NEURALWATT_API_KEY` set for this run only:
   - **No shared-singleton symptoms and no rate-limit errors of any kind** — the overnight
     failure mode (`AssertionError("llm must be set to compute score")` / mass 429s) did not
     recur anywhere in either the full 40-row attempt or a faster 5-row slice. This is the key
     evidence the M1 fix works.
   - A **full 40-row run was attempted and did not finish within ~67 minutes** before being
     killed for diagnosis — **not an infinite hang**. Its progress output showed continuous,
     correct forward progress the entire time: `glm-5.2-short` and `kimi-k2.7-code` both
     completed their full 40/40 rows cleanly in ~28-29 minutes each (with a handful of isolated
     `TimeoutError` exceptions on individual rows, each correctly absorbed as a NaN for that row
     rather than crashing the whole model — this is `S3`'s mitigation and ragas's own per-job
     error handling both working as intended). `qwen3.5-397b`, however, was still only 26/40
     rows in after 65 minutes at a consistent ~150s/row — independently confirming the
     `NEURALWATT_MODELS` catalog's own note and this task's own prediction that this specific
     model "has a genuinely high real-world failure rate."
   - **Root cause of the slow wall-clock time (not a bug, an accepted design tradeoff):** the
     confirmed-safe `RunConfig(max_workers=1)`-per-judge-thread design (see M2 above) gives each
     judge model exactly ONE dedicated concurrency slot for its entire 40-row sequence, for the
     whole run. A model that is persistently slow/timing-out (`qwen3.5-397b`) cannot borrow the
     other two judges' slots once they finish early, so the total wall-clock time is bounded by
     the SLOWEST model's fully-serial 40-row time (projected ~100 minutes) rather than by the
     panel's 3-way parallelism. This is an inherent consequence of the exact design mandated
     above to safely respect NeuralWatt's confirmed 3-slot account-wide limit — a more
     sophisticated design (e.g. a shared `threading.Semaphore(3)` across all 3 judges' row-level
     calls instead of one fixed slot per judge) would let fast judges' freed-up capacity help
     the slow one, but that is a real scope increase beyond this task's fix list and was not
     built here.
   - A **5-row slice completed cleanly in 714.5s (~11.9 min)**, with **0% NaN across all 3
     NeuralWatt models** (`glm-5.2-short`, `kimi-k2.7-code`, `qwen3.5-397b` all returned 5/5
     valid faithfulness scores, mean scores 0.586 / 0.543 / 0.509 respectively) and correct
     `delta_m`/`high_disagreement`/`suspiciously_unanimous`/`all_judges_failed` fields on every
     row of real output — confirming the S5/S6 fixes end-to-end against live data, not just the
     isolated unit check in step 2.
   - No comparison against the overnight run's saved
     `/tmp/nfcorpus_eval_v2/neuralwatt_multijudge_tuned_r1.json` was possible for the full
     40-row file (this run did not complete it), but that file's own `_meta.elapsed_sec` was
     ~3597s (~60 min) for the SAME file under the old serialized workaround — comparable in
     order of magnitude to this run's projected ~100 minutes, i.e. **not the "meaningfully
     faster" outcome hoped for**, specifically because of `qwen3.5-397b`'s persistent slowness
     interacting with the fixed one-slot-per-judge design (see above). The two well-behaved
     judges DID run genuinely 3x-parallel-fast relative to their own row counts (~28-29 min for
     40 rows each, running concurrently with each other and with the slow judge, not
     sequentially after it) — the panel-level parallelism itself is confirmed working; it's
     specifically bounded by the worst-case judge, as intrinsic to this design.
4. Confirmed no unintended network calls beyond the deliberate verification runs above (one
   killed 40-row attempt, one completed 5-row run, one further 5-row run after the fallback
   addition below), and confirmed `run_multi_judge_faithfulness_crosscheck()`'s concurrency
   model was NOT changed (remains sequential; only its M1 shared-singleton bug was fixed).

### Follow-on fix added mid-verification: Ollama-cloud fallback for failed NeuralWatt judges

The live verification above directly observed real `TimeoutError` instability on NeuralWatt
(the isolated per-row failures on `glm-5.2-short`/`kimi-k2.7-code`, and `qwen3.5-397b`'s
near-total slowness). In response, `_score_one_model()` now attempts **one retry via the
equivalent Ollama-hosted cloud model** before reporting a NeuralWatt judge as failed, using a
new mapping constant near `NEURALWATT_MODELS`:

```python
_NEURALWATT_TO_OLLAMA_FALLBACK = {
    "glm-5.2-short": "glm-5.2:cloud",
    "kimi-k2.7-code": "kimi-k2.7-code:cloud",
    "qwen3.5-397b": "qwen3.5:397b-cloud",
}
```

The fallback reuses the existing `_ollama_judge_llm()` builder (no duplicated client
construction) and the same M1 fresh-`Faithfulness()`-per-call fix, with its own run_config
(`_OLLAMA_JUDGE_MAX_WORKERS`, since Ollama isn't subject to NeuralWatt's account-wide limit).
The per-model result dict gained a `"judge_used"` field (`"neuralwatt"` | `"ollama_fallback"` |
`None`) so the aggregation/reporting layer can distinguish "NeuralWatt succeeded outright" from
"had to fall back to Ollama" — this matters for the cross-provider reliability comparison the
panel exists to make. On a successful fallback, the original NeuralWatt failure is preserved in
`"neuralwatt_error"`/`"neuralwatt_error_type"` rather than discarded, so the fallback never
silently hides which provider actually had the problem. The CLI's
`--neuralwatt-multi-judge-crosscheck` output was updated to print a one-line note when a model's
score came from the fallback rather than NeuralWatt directly.

**Re-verified after adding the fallback:** `python3 -m py_compile scripts/evaluate_ragas.py`
still passes; a second 5-row live run against the same slice completed successfully (see the
JSON output for exact scores/timing) with the refactored `_score_one_model()`/shared
`_run_faithfulness()` helper behaving identically to the pre-fallback version when NeuralWatt
itself succeeds (the fallback path was not exercised in this particular run since all 3
NeuralWatt judges succeeded on this slice — its correctness rests on code review and the shared
`_run_faithfulness()` helper being identical to the already-verified primary path, not on a
live-triggered fallback).

A full 40-row re-run was NOT performed after this addition, given the ~60-100 minute wall-clock
cost demonstrated above for that file size and that the fallback logic's own correctness does
not depend on scale.

## 2026-07-16 — NeuralWatt multi-judge consensus with geometric-median aggregation added (`run_neuralwatt_multi_judge_consensus()`, NOT yet run against real data)

**What this adds:** `run_neuralwatt_multi_judge_consensus(csv_path, model_names=["glm-5.2-short",
"kimi-k2.7-code", "qwen3.5-397b"], concurrency=3, disagreement_threshold=0.3,
unanimous_tolerance=0.05)` in `scripts/evaluate_ragas.py`, plus a `--neuralwatt-multi-judge-crosscheck
CSV_PATH` CLI flag mirroring `--multi-judge-crosscheck`. This is the NeuralWatt counterpart to the
existing `run_multi_judge_faithfulness_crosscheck()` (local Ollama judges), extended with the
consensus-aggregation logic that function's docstring had documented as a deliberately-deferred
future extension.

**Why:** a single-judge faithfulness score — or even three independent judges reported side by
side with no aggregation — leaves a reviewer to eyeball disagreement manually and gives no
principled way to combine three numbers into one, nor any signal when all three judges agree for
the wrong reason. This closes that gap using patterns from prior research already cited in this
project (PoLL, arXiv:2404.18796; RoPoLL, arXiv:2606.30931; Kohli 2026, arXiv:2605.29800; the
ProofAgent Harness "Consensus Agent" pattern).

**What it does, per question:**
1. Runs the faithfulness metric independently through each of the 3 NeuralWatt judge models
   (`_neuralwatt_llm()`), in parallel via `concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)`
   (genuinely new — the existing Ollama `run_multi_judge_faithfulness_crosscheck()` actually runs its
   per-model loop sequentially today, despite its docstring noting the per-model try/except was
   structured to make later parallelization easy).
2. Aggregates the 3 scores via **geometric median** (`_geometric_median()`), implemented directly as
   Weiszfeld's iterative algorithm (Weiszfeld, 1937; Vardi & Zhang, PNAS 97(4), 2000) — not a mean or
   median shortcut mislabeled as "geometric median". For scalar (1-D) inputs the geometric median is
   mathematically identical to the ordinary median (a real property, not an approximation), which the
   unit test below confirms directly by checking the result differs from the mean.
3. Computes the spread `delta_m = max(scores) - min(scores)` per question and flags rows
   `delta_m > disagreement_threshold` (default 0.3) as **"high_disagreement"**.
4. ALSO flags rows with `delta_m < unanimous_tolerance` (default 0.05) as **"suspiciously_unanimous"**
   — per Kohli 2026, unanimous agreement across judges can reflect correlated bias/a shared blind spot
   rather than extra reliability, so it is surfaced as a signal to check, not silently trusted.
5. For "high_disagreement" rows only, calls a new `_claude_tiebreaker_faithfulness()` helper (reusing
   this module's existing `_default_judge_llm()` machinery, no new prompt/client path) to get a 4th
   independent opinion, so the returned row carries all 4 individual scores (3 NeuralWatt + Claude)
   plus the geometric median and both flags.

**Shared-logic refactor:** factored the CSV -> `EvaluationDataset` loading step (parsing the
stringified `retrieved_contexts` column, building `SingleTurnSample`s) out of
`run_multi_judge_faithfulness_crosscheck()` into a new `_load_faithfulness_dataset_from_csv()`
helper, reused by both functions — this was the only genuinely shared logic between the two; the
surrounding flow (sequential vs. parallel, no aggregation vs. geometric-median + flagging + Claude
tiebreaker) differs enough that merging further would have been a premature abstraction.

**`scipy` dependency — deliberately NOT added:** the task that motivated this asked for the
geometric median to be computed via `scipy.optimize`/`scipy.stats` "or a small Weiszfeld iteration".
`_geometric_median()` uses a pure-Python Weiszfeld iteration with no `scipy` import, so no new
dependency was needed and none was added to `requirements.txt` — adding an explicitly-unused
dependency would have been misleading. (Note `scipy` is already available transitively via
`scikit-learn>=1.5.0`, already in `requirements.txt` for `scripts/visualize_embeddings.py`, if a
future change genuinely needs it directly.)

**Verification performed (implementation only, per this task's explicit scope):**
- `python3 -m py_compile scripts/evaluate_ragas.py` — passes.
- A standalone inline unit test of `_geometric_median()` alone (algorithm copied out, no `ragas`
  import needed since `ragas` isn't installed in this environment) confirmed: three identical
  values return that value; `[0.1, 0.5, 0.9]` returns `0.5` (the median); `[0.0, 0.2, 1.0]` returns
  `0.2` (the median), explicitly NOT `0.4` (the mean) — confirming the implementation is not
  secretly just averaging; a single value returns unchanged; empty input raises `ValueError`.
- **NOT run:** no live NeuralWatt API call, no live Anthropic/Claude tiebreaker call, and no
  `NEURALWATT_API_KEY` was set anywhere in this work. `run_neuralwatt_multi_judge_consensus()` is
  implementation-complete but unexercised against real data, the same status
  `run_multi_judge_faithfulness_crosscheck()` carries above.

## 2026-07-16 — Generation-side verify-then-repair step added (`verify_and_repair_answer()`, OFF by default)

**What this adds:** `verify_and_repair_answer(question, contexts, draft_answer)` in
`scripts/evaluate_ragas.py`, plus a `verify_and_repair: bool = False` parameter on
`generate_answer()` and a `--verify-repair` CLI flag (only takes effect together with
`--generate`). It runs as an optional final step on `generate_answer()`'s own output:

1. Extracts every `[C]`-tagged claim from the draft answer (`_extract_c_tagged_claims()` — reuses
   the existing blended `[C]`/`[G]` tagging format from the attribution-tagged prompt; no new
   tagging scheme).
2. If there are no `[C]`-tagged claims, returns the draft unchanged — nothing to verify.
3. Batches **all** claims from that one answer into a **single** verification LLM call (never one
   call per claim) asking a cheap/fast judge (`claude-haiku-4-5` on the direct API; the same
   generation model on the Vertex path, since the deployed Vertex model may not include Haiku)
   whether each claim is actually directly traceable to the retrieved context, vs. merely
   topically related or embellished/synthesized beyond it.
4. For every claim the judge flags as unsupported, applies one of two repairs per the judge's own
   classification: **retag** (`[C]` -> `[G]`) when the claim is accurate general-knowledge content
   that was simply mistagged, or **rewrite** (replace with a tighter, more literal extraction the
   judge supplies, still tagged `[C]`) when the claim drew on context but added specifics/synthesis
   not actually present.
5. Fails safe: any error in the verification call (network failure, unparseable judge response)
   is caught, logged, and returns the draft answer unmodified rather than raising or corrupting it.

**Cost:** exactly one extra LLM call per `generate_answer()` invocation when the draft has at
least one `[C]`-tagged claim (zero extra calls for fully `[G]`/untagged answers) — all claims in
that answer are batched into the one call regardless of count.

**Why this, specifically:** this is the direct, previously-unimplemented other half of the
Self-Correcting RAG finding cited in the context-review entry below. That paper's ablation
(arXiv:2604.10734) found that cleaning up retrieval/context **alone** left faithfulness flat, but
that adding a review-and-correction step on the **generated output** (not the input) measurably
fixed faithfulness — their AP metric moved 0.58 -> 0.85. This project's own A/B test of the
blended `[C]`/`[G]` prompt found exactly the predicted gap: the blended prompt fixed the
75%-refusal-rate / `answer_relevancy` problem but caused a real faithfulness regression (-0.30 to
-0.35 absolute, p<0.001) relative to the old strict-refusal prompt. Root-causing that regression
(manual inspection of low-scoring samples) attributed most of it to prose **style** around
`[C]`-tagged claims — markdown headers, expository/synthesized framing that lightly embellishes a
context-derived fact rather than extracting it near-verbatim — tripping RAGAS's sentence-level
faithfulness judge, not the underlying content actually being ungrounded. `verify_and_repair_answer()`
targets that style-driven failure mode directly, at the one place (the generated output) the paper's
ablation says actually moves the needle, without reverting to the strict-refusal prompt and
reintroducing the refusal-rate regression.

**Off by default, pending a real A/B run:** `verify_and_repair` defaults to `False` on
`generate_answer()` and the CLI flag is opt-in, so existing behavior (and any existing
tests/callers) is unaffected unless explicitly enabled. A rigorous before/after comparison run
(the same kind of statistical rigor pass this project has already applied to the blended-prompt
change) is the natural next step to quantify how much faithfulness this actually recovers on a
real testset, and is explicitly **out of scope** for this change — this entry only covers
shipping the capability and confirming it behaves as designed on hand-built examples.

**Verified live (hand-built examples, not a full statistical run):** ingested a small test corpus
(hierarchical chunking / RRF / Qdrant) into a throwaway Qdrant collection and ran a real
retrieval + `generate_answer()` call for a question that produced a mix of `[C]`- and
`[G]`-tagged content. On that example, every `[C]`-tagged claim in the draft answer was already a
close, well-attributed extraction, and `verify_and_repair_answer()` correctly left the answer
untouched — a valid outcome, not a null result, since the verification step is supposed to be a
no-op on already-well-grounded answers. To confirm the repair path itself fires when it should,
also ran `verify_and_repair_answer()` directly against a hand-built draft answer containing one
faithful `[C]` claim and one deliberately fabricated `[C]` claim (an invented "23% improvement"
statistic not present anywhere in the context): the faithful claim was left unchanged and the
fabricated claim was correctly flagged as unsupported and rewritten to remove the fabricated
number. Deleted the throwaway Qdrant collection afterward.

## 2026-07-16 — Optional CRAG-style context review/distillation step added (`retrieval.context_review_enabled`, OFF by default)

**What this adds:** a new optional step, `review_and_distill_context()` in `scripts/utils.py`,
implementing the "decompose-then-recompose" pattern from Corrective RAG (CRAG, arXiv:2401.15884).
When enabled, it runs at the very end of `EfficientRAG.retrieve()` — after cross-encoder
reranking and parent-chunk expansion, on the final hits that would otherwise go straight to
`generate_answer()`. For each hit: splits `text` into sentences (`_split_into_sentences()`, a
simple regex heuristic with a common-abbreviation guard — no new NLP dependency), scores every
sentence's relevance to the query in ONE batched LLM call per hit (`_score_sentence_relevance()`
— not one call per sentence, to keep this affordable), discards sentences scoring below
`retrieval.context_review_threshold` (default `0.3`, lenient by design — biases toward keeping
borderline sentences since losing a genuinely relevant one costs `context_recall` more than
keeping a mildly noisy one costs `context_precision`), and recomposes the survivors back into a
denser `text`. Every other field on the hit (`score`, `source`, `parent_id`, `chunk_type`,
`tags`, `summary`, `child_text`) passes through unchanged. Reuses
`evaluate_ragas.py`'s `_build_anthropic_client()` for credential/client selection (direct API vs.
Vertex AI) rather than duplicating that logic, imported lazily to avoid a circular import with
`utils.py`. Always uses a cheap/fast Haiku-class model for the scoring call
(`claude-haiku-4-5` direct, `claude-haiku-4-5@20251001` on Vertex, both overridable) regardless of
which model the main generation path uses — this runs once per retrieved hit on every query, so
it needs to be cheap, not high-reasoning.

**New config keys** (`config/config.yaml`, `load_config()`'s defaults in `scripts/utils.py`):
`retrieval.context_review_enabled` (default `false`) and `retrieval.context_review_threshold`
(default `0.3`). **OFF by default** — this is an experimental addition, not proven to help
end-to-end yet, so enabling it requires an explicit opt-in and existing behavior is completely
unchanged when it's left off (verified: `EfficientRAG.retrieve()` only calls
`review_and_distill_context()` when the flag is true).

**Why:** noise in retrieved context — off-topic sentences that ride along inside a hit merely
because they share a parent chunk or a loosely-related child chunk with genuinely relevant
content — is a distinct problem from anything else tracked in this project so far, and CRAG's
decompose-then-recompose pattern targets it directly: strip the noise sentence-by-sentence before
the context ever reaches the prompt. The expected benefit is **`context_precision`** specifically
(less irrelevant material diluting the retrieved set), with a possible secondary/indirect benefit
to `context_recall` and `answer_relevancy` from a denser, more focused context — not a
faithfulness fix.

**Explicit caveat — this is NOT expected to fix the faithfulness regression documented earlier in
this file / in `scripts/evaluate_ragas.py`'s module docstring (the "attribution-tagged blending"
policy).** That regression is caused by `generate_answer()`'s own prompt design: it deliberately
lets the final generator supplement thin/tangential context with general knowledge (tagged
`[G]`), which is what suppresses the faithfulness score — not noise in the retrieved chunks
themselves. Cleaning the context can't change a generator-side policy choice. This distinction
was checked against a real ablation in a separate paper, **Self-Correcting RAG (arXiv:2604.10734)**,
which found that context-side cleaning alone left faithfulness flat. Do not read an unmoved
faithfulness score after enabling `context_review_enabled` as "it didn't work" — check
`context_precision` (and secondarily `context_recall`/`answer_relevancy`) instead, since that is
the axis this feature actually targets. Both this note and the full reasoning are also captured
in `review_and_distill_context()`'s docstring in `scripts/utils.py` so it isn't lost in this file
alone.

**Verified live:** ingested a small 5-paragraph test corpus into a throwaway Qdrant collection,
deliberately interleaving on-topic paragraphs (hierarchical chunking, cross-encoder reranking,
binary/scalar quantization) with two clearly off-topic tangents (a paragraph about Lisbon
tourism, a paragraph about pineapple on pizza). Hierarchical chunking merged several paragraphs
into one parent chunk (as expected given `parent_chunk_size`), so the reranking-relevant parent
hit returned by `retrieve()` for the query "How does cross-encoder reranking improve retrieval
pipelines?" contained all four topics mixed together. With `context_review_enabled: false`
(baseline), that hit's `text` was the full 1697-character mixed-topic parent chunk. With
`context_review_enabled: true`, the same hit's `text` shrank to 530 characters containing only
the three reranking-relevant sentences — the hierarchical-chunking, Lisbon, and pizza sentences
were all correctly discarded. A separate hit whose parent chunk was already single-topic
(quantization) was left unchanged (426 -> 426 chars), confirming the step doesn't damage
already-clean context. Deleted the throwaway Qdrant collection afterward.

## 2026-07-16 — Judge-independent human review tier added (`--export-human-review` / `--summarize-human-review`)

**Motivation:** every quality signal in this project so far comes from an LLM judge — RAGAS's
five metrics, the closed-book comparison arm, the Claude web_search comparison arm, and the
multi-judge Ollama faithfulness cross-check (see the entry below). This session hit repeated
judge-reliability problems in practice: a Gemini-based second judge was tried and explicitly
**rejected by the user** for being a stale model version, and a separate statistical rigor pass
found several RAGAS metric deltas were **not distinguishable from judge noise** at reasonable
sample sizes. Adding yet another LLM judge doesn't close that gap — it's still an LLM judging LLM
output. This adds the one signal that isn't: a human manually labeling a sample of real
(question, retrieved_contexts, generated_answer) triples.

**What shipped**, all in `scripts/evaluate_ragas.py`:

- `HUMAN_REVIEW_LABELS` / `HUMAN_REVIEW_RUBRIC` — the canonical three-label rubric (`Grounded`,
  `Partially grounded`, `Not grounded`), defined once and referenced everywhere (module
  docstring, CSV header comment, `docs/HUMAN_REVIEW_RUBRIC.md`) so it can't drift into multiple
  versions. `Partially grounded` is explicitly tied to `generate_answer()`'s existing `[C]`/`[G]`
  attribution tags — a real, checkable correlation, not just a naming coincidence.
- `export_for_human_review(testset, rag_retrieve_func, llm_generate_func, output_csv_path)` —
  reuses the exact same `_make_rag_retrieve_func()` / `generate_answer()` machinery
  `run_ragas_evaluation()` uses (no duplicated retrieval/generation logic) and writes a CSV with
  `question`, `retrieved_contexts` (joined, truncated to 4000 chars for spreadsheet readability),
  `generated_answer`, and empty `human_label` / `human_comment` columns for a reviewer to fill in.
  The CSV's first line is a `#`-prefixed comment restating the three valid label strings.
- `summarize_human_review(csv_path, ragas_csv_path=None)` — reads a filled-in copy of that CSV,
  prints a label count/percentage breakdown (flagging any `human_label` value that doesn't
  exactly match the rubric's three strings, e.g. wrong case), and — if a saved RAGAS
  per-question results CSV is also passed — cross-tabulates mean RAGAS scores by human label and
  explicitly checks the rubric's own falsifiable prediction: mean `faithfulness` for `Partially
  grounded` rows should be lower than for `Grounded` rows. Prints "PASSED" or "FLAGGED"
  accordingly, rather than just dumping a table and letting the reader guess.
- New CLI flags: `--export-human-review <output_csv>` (requires `--testset` + `--collection`) and
  `--summarize-human-review <filled_csv>` with optional `--compare-ragas <ragas_csv>`.
- New doc: `docs/HUMAN_REVIEW_RUBRIC.md` — the rubric in full, why it exists, and usage.

**Verified live:** ingested a tiny 3-paragraph test document (hierarchical chunking / RRF /
Qdrant) into a throwaway Qdrant collection, ran `--export-human-review` against a 3-question
testset (two on-topic, one deliberately off-topic — a Mars-rover question with no relevant
context in the corpus) via Claude on Vertex AI. Confirmed real retrieval + generation ran and
produced a readable CSV: the off-topic question correctly triggered a fully `[G]`-tagged,
"general-knowledge-based" answer, and the two on-topic questions produced mixed `[C]`/`[G]`
answers — exactly the label variety the rubric is meant to distinguish. Hand-filled the CSV with
`Grounded` / `Partially grounded` / `Not grounded` labels plus comments and ran
`--summarize-human-review`: label distribution printed correctly (1/1/1, 33.3% each). Built a
matching fake RAGAS results CSV (`faithfulness` 0.91 / 0.62 / 0.10 for the three rows) and ran
`--summarize-human-review --compare-ragas`: the cross-tab printed correctly and the sanity check
correctly reported "PASSED" (`Partially grounded` mean faithfulness 0.620 < `Grounded` mean
faithfulness 0.910). Also verified the typo-detection path by lowercasing one label
(`"partially grounded"`) and confirming it's flagged as non-rubric text rather than silently
miscounted. Deleted the throwaway Qdrant collection afterward.

## 2026-07-16 — New diagnostic tool: `scripts/visualize_embeddings.py` (embedding-collapse visualization)

**Motivation:** While diagnosing the "retrieval returns the same generic chunk regardless of
query" symptom described further below in this document, the root cause turned out to be the
`--recursive` ingest data-loss bug (Bug 1/2), not embedding collapse — but confirming that took
live debugging: pulling raw vectors out of Qdrant and computing pairwise cosine similarity by
hand to rule embedding collapse in or out. A standalone diagnostic that makes an *actual*
embedding-collapse scenario (all chunks landing on nearly the same vector — e.g. from a broken
embedding call, a frozen/misconfigured model, or every chunk accidentally embedding the same
text) visually and numerically obvious up front would have shortened that diagnosis. This is a
new capability, not a fix for a prior bug.

**What it does:** `scripts/visualize_embeddings.py` is a standalone CLI (reuses
`load_config()`/`EfficientRAG` from `scripts/utils.py`, no reimplemented Qdrant connection logic)
that:

- Pulls all (or a `--limit`-capped sample of) dense vectors + payloads for CHILD chunks
  (`chunk_type=child`) from a given collection via `qdrant.scroll(..., with_vectors=True,
  with_payload=True)`, using the same `Filter`/`FieldCondition`/`MatchValue` pattern
  `EfficientRAG.retrieve()` already uses to exclude zero-vector parent points from search —
  parent points carry a dummy all-zero `"dense"` vector (see the parent-chunk fix further below)
  and would otherwise show up as a misleading cluster-at-the-origin artifact.
- Projects the vectors to 2D via **both** PCA (scikit-learn, printing explained variance ratio for
  the first 2 components) and UMAP (umap-learn), controllable via `--method pca|umap|both`
  (default `both`). UMAP's `n_neighbors` is scaled down to `min(15, n_points - 1)` instead of
  using the library default of 15 unconditionally — UMAP's defaults are tuned for large corpora
  and produce misleading layouts (or outright errors) on small point counts, which matters for
  this project's frequent small-corpus smoke tests.
- Colors points by the `source` payload field with a legend, so clustering-by-document is
  visually checkable at a glance.
- Saves plot(s) to PNG file(s) under a `--output` prefix (default `embeddings_viz`) via
  matplotlib's non-interactive `Agg` backend — this is a CLI diagnostic tool, not a notebook, so
  it never attempts inline display, only saves to disk and prints the save path.
- Independently of the visualization, computes and prints a **pairwise cosine similarity
  degeneracy check**: mean cosine similarity across a random sample of points, plus the fraction
  of pairs above a `--sim-threshold` (default `0.98`), flagging a plain-text warning when the
  numbers suggest embedding collapse. This gives a numeric signal even if the 2D projection itself
  is ambiguous or misleading.

**Dependencies added:** `scikit-learn`, `umap-learn`, `matplotlib` (all `>=`-pinned per this
project's non-langchain dependency convention — no conflict forced a tighter pin). Installed
alongside the existing `ragas`/`langchain-*`/`langchain-ollama` stack and verified clean with a
live `pip check` (0 broken requirements) in this environment.

**Verified live:** ingested a small 3-topic test corpus (astronomy, baking, chess — 3 files, 12
child chunks total) into a throwaway collection and ran the script with default `--method both`.
Both `_pca.png` and `_umap.png` were produced (~54KB and ~53KB respectively — non-trivial file
sizes, not empty/blank plots), no exceptions were raised, and UMAP correctly logged
`n_neighbors=11` (scaled down from the default 15 for this 12-point corpus, confirming the small-
corpus scaling logic engaged). PCA reported explained variance ratio `0.2543, 0.2264` (total
`0.4807`) for the first 2 components. The degeneracy check reported a mean pairwise cosine
similarity of `0.5500` across all 66 possible pairs (well under the full sample cap, so no
sub-sampling was even needed at this corpus size) with `0.00%` of pairs above the `0.98`
threshold — a healthy spread consistent with three genuinely distinct topics, correctly **not**
triggering the collapse warning. Also verified `--method pca` alone correctly skips the UMAP step
and its PNG. The test collection was deleted afterward.

**If you want to reproduce this or extend it:** the degeneracy check
(`compute_degeneracy_check()`/`print_degeneracy_report()`) is independent of the plotting code and
can be reused standalone if a future tool wants just the numeric check without matplotlib. The
child-chunk vector fetch (`fetch_child_vectors()`) is the reusable piece for any future diagnostic
that needs raw dense vectors + payloads without reimplementing the scroll/filter logic.

## 2026-07-16 — Hierarchical chunking switched from character-based to token-based sizing

**Symptom:** `_hierarchical_chunk()`'s `recursive_split()` sized chunks by comparing raw
`len(t)` (character count) against `child_chunk_size`/`parent_chunk_size` from
`config/config.yaml`. But the embedding model (`BAAI/bge-small-en-v1.5`) and any LLM
consuming chunk text have TOKEN limits, not character limits, and chars-per-token varies
with content density — dense technical text runs noticeably fewer chars/token than plain
prose (measured ~4.5-5.3 chars/token on sample English text through this model's
tokenizer). Character-based sizing was therefore only an approximation of the thing that
actually matters, with no fixed conversion factor.

**Fix:** `EfficientRAG._token_len(text)` (new method in `scripts/utils.py`) counts tokens
using the embedding model's own HuggingFace tokenizer — `self.embed_model.tokenizer`,
confirmed live to be a `transformers.BertTokenizer` for `BAAI/bge-small-en-v1.5`, exposing
a normal `.encode()`/`.decode()` API — via
`len(self.embed_model.tokenizer.encode(text, add_special_tokens=False))`. No new
dependency (e.g. tiktoken) was needed since the `SentenceTransformer` object already
carries the exact tokenizer the embedding step uses.

`_hierarchical_chunk()`'s `recursive_split()` now uses `_token_len()` everywhere it
previously used `len()`:
- The initial "is this already small enough" check.
- The per-part accumulation check when packing split parts into a chunk under the size
  budget — reworked to tokenize each `part + sep` **once** up front and track a running
  token-count total while accumulating `current`, rather than re-tokenizing the whole
  `current` string on every part (the naive character-to-token translation). This keeps
  tokenizer calls O(num_parts) instead of O(num_parts²), which matters for large documents.
- The last-resort fallback (no separator found at all) now encodes the remaining text once
  and slices token-id windows directly (decoding each window back to text), instead of
  slicing raw characters.

**Default value changes:** `child_chunk_size`/`child_chunk_overlap`/`parent_chunk_size` in
`config/config.yaml` (and `load_config()`'s defaults in `scripts/utils.py`) changed from
`400`/`50`/`1800` (characters) to `100`/`20`/`400` (tokens). These are NOT the same numbers
relabeled — re-using `400` as a token count would have made child chunks ~4-5x larger than
originally intended (a "small precise-retrieval chunk" ballooning into a substantial
fraction of the model's 512-token max). The new defaults were derived by dividing the old
character values by the measured ~4.5 chars/token average, preserving the original
*intended* granularity (small child chunks for precise retrieval, parent chunks large
enough for full context) rather than the old numbers.

**Verified live:** ran `_hierarchical_chunk()` on a real multi-paragraph markdown test
document (4 sections, ~2600 characters) and measured every resulting chunk's token count
with the same tokenizer: 8 child chunks ranged 51-85 tokens (all ≤ the configured
`child_chunk_size: 100`), 2 parent chunks were 191 and 358 tokens (both ≤ the configured
`parent_chunk_size: 400`) — no chunk exceeded its configured token budget, and none were
degenerately tiny. Also ran a full `ingest()` through a throwaway Qdrant collection
(`token_chunk_test_collection`): reported `child_points_ingested: 8, parent_points_ingested:
2` with no errors, confirming the token-based chunks flow through embedding/upsert
correctly. The collection was deleted after verification (`qdrant.delete_collection(...)`).
A `Token indices sequence length is longer than the specified maximum sequence length ...
(549 > 512)` warning observed during this test comes from the separate fastembed BM25
sparse-embedding model's own initialization (it prints before any per-chunk processing, and
every actual chunk's bge-tokenizer length was confirmed well under 512) — unrelated to this
change.

**If you see this again:** check `EfficientRAG._token_len()` and the `recursive_split()`
inner function inside `_hierarchical_chunk()` in `scripts/utils.py` for `tok_len(...)` calls
in place of raw `len(...)`. Confirm `config/config.yaml`'s `indexing.child_chunk_size` /
`child_chunk_overlap` / `parent_chunk_size` comments still say TOKEN counts, not characters.

## 2026-07-16 — `_fetch_url()` article extraction upgraded to trafilatura (boilerplate leaking into RAG chunks)

**Symptom:** URL ingestion produced chunks polluted with page boilerplate that wasn't real article
content — related-links blocks, comment sections, and promotional callouts were surviving
extraction and ending up embedded and searchable alongside genuine article text.

**Root cause:** `EfficientRAG._fetch_url()` in `scripts/utils.py` used plain `requests.get()` +
BeautifulSoup, stripping only a fixed list of tags (`script`, `style`, `nav`, `footer`, `header`,
`aside`) before taking the remaining text (capped at 50,000 characters). That fixed tag list
catches structural chrome but has no way to recognize boilerplate that isn't wrapped in one of
those specific tags — related-links widgets, comment sections, and promo callouts are frequently
just `div`s or `section`s with no distinguishing tag, so they passed straight through.

**Fix:** `_fetch_url()` now tries `trafilatura.extract()` first — a purpose-built readability-style
extraction library (added to `requirements.txt` as `trafilatura>=1.8.0`) that scores content blocks
on structural and textual signals rather than a fixed tag denylist, and benchmarks substantially
better than BeautifulSoup tag-stripping at isolating real article content. If `trafilatura.extract()`
returns `None` (some pages genuinely fail extraction either way), `_fetch_url()` falls back to the
original BeautifulSoup tag-stripping approach, so ingestion never hard-fails on a URL the old method
could at least partially handle. The 50,000-character cap is unchanged.

**Verified live:** ingested a real Wikipedia article through the trafilatura path and confirmed zero
navigation-chrome leakage — checked for "Jump to content", "Main menu", "Toggle", and similar
Wikipedia-chrome strings, all absent from the extracted text. Also forced `trafilatura.extract()` to
return `None` via mocking and confirmed the BeautifulSoup fallback path still returns non-empty,
usable text.

**If you see this again:** check `EfficientRAG._fetch_url()` in `scripts/utils.py` for the
`trafilatura.extract()` call and its `None`-triggered fallback to the BeautifulSoup tag-stripping
block. Confirm `trafilatura` is still pinned in `requirements.txt`.

**Design detour (rejected approaches, recorded so they aren't re-proposed):** before landing on the
trafilatura swap above, a query-time router between local Qdrant retrieval and the separate
`research-tool` MCP was considered and **rejected** — the two tools are independent and it's
Hermes/the calling agent's job to decide which one to invoke, not this skill's job to fuse them
internally. A follow-on proposal to directly import `research-tool`'s internal `ApifyProvider`
class was also **rejected** — importing a sibling tool's internal, non-published class creates
unwanted coupling between otherwise-independent tools ("distributed monolith" risk). The actual fix
that shipped is the much smaller trafilatura swap described above.

## 2026-07-16 — Multi-judge Ollama faithfulness cross-check added (Gemini rejected as second judge)

**Context:** A second-opinion judge for the `faithfulness` metric was wanted, since a single
Claude judge can't distinguish "the answer is genuinely unfaithful" from "this judge is
miscalibrated on this corpus." Gemini was tried first as the obvious second-vendor candidate and
was explicitly **rejected by the user**: the readily available Gemini model version was stale
enough to make it an unreliable comparison point, not worth the added complexity.

**Design chosen instead:** a multi-judge cross-check using three independent-family cloud models
served locally via Ollama — `glm-5.2:cloud`, `kimi-k2.7-code:cloud`, and `minimax-m3:cloud` — so
the cross-check doesn't depend on any single vendor's judge quirks, and doesn't require managing
another cloud API key/quota. Implemented in `scripts/evaluate_ragas.py`:

- `_ollama_judge_llm(model_name, num_ctx=200000)` — builds
  `LangchainLLMWrapper(ChatOllama(model=model_name, base_url=..., num_ctx=...))`. The `num_ctx`
  parameter name was verified against the installed `langchain-ollama==0.3.10` source
  (`langchain_ollama/chat_models.py`: `num_ctx: Optional[int] = None`, "Sets the size of the
  context window used to generate the next token.") rather than guessed.
- `run_multi_judge_faithfulness_crosscheck(csv_path, model_names=[...])` — reads a previously
  saved per-question RAGAS results CSV, rebuilds `SingleTurnSample`/`EvaluationDataset` the same
  way `run_ragas_evaluation()` does, and re-scores `faithfulness` only, once per Ollama model,
  returning a dict keyed by model name so failures are isolated per-model.
- New CLI flag `--multi-judge-crosscheck <csv_path>` prints a comparison table: the CSV's original
  (Claude) `faithfulness` column alongside each of the three Ollama judges' scores.

**Dependency note:** this required installing `langchain-ollama`, which surfaced (and then fixed)
a venv breakage — see the "langchain-ollama pin" note in `requirements.txt`'s own comments for the
`langchain-core` version-conflict details and the exact pin (`langchain-ollama==0.3.10`) that
resolves it without touching the existing `langchain-anthropic`/`langchain-openai`/
`langchain-google-vertexai` pins.

**Status: implemented but intentionally NOT YET RUN.** This capability requires Ollama running
locally with all three models pulled (`ollama list` should show `glm-5.2:cloud`,
`kimi-k2.7-code:cloud`, `minimax-m3:cloud`), which is a separate manual step the user explicitly
wanted to gate ("I'll tell you when you can run it"). No `ragas.evaluate()` call, `ChatOllama`
`.invoke()`, or network request to Ollama was made while implementing this — verification was
limited to `py_compile`/`--help`/import-only checks.

## 2026-07-16 — Strict-refusal generation prompt replaced with attribution-tagged blending

**Symptom:** On a hard corpus, live RAG evaluation runs showed the RAG pipeline scoring far
*worse* on `answer_relevancy` than a no-retrieval closed-book baseline (see
`generate_closed_book_answer()` / `run_closed_book_evaluation()`) — despite retrieval itself
(context_precision/context_recall) looking reasonable. Manually inspecting generated answers
showed the model refusing to answer outright roughly 75% of the time whenever the retrieved
context was thin, tangential, or only partially relevant to the question.

**Root cause:** `generate_answer()`'s prompt told the model to "Answer the question using ONLY
the information in the provided context. If the context does not contain the answer, say so
explicitly." On a hard corpus, retrieval frequently surfaces context that's related but doesn't
fully answer the question — and the strict wording pushed the model to refuse rather than give a
partial, still-useful answer. A refusal is never a relevant answer, so `answer_relevancy` cratered
even on questions where the retrieved context contained real, usable signal. This made the RAG
arm look strictly worse than not retrieving anything at all, which is the opposite of what
retrieval is supposed to buy you.

**Fix:** `generate_answer()`'s prompt now asks the model to blend retrieved context with its own
general knowledge, with inline attribution tags so the source of each claim stays auditable:

- Use CONTEXT wherever it applies, tagging each context-derived claim inline with `[C]`.
- If CONTEXT is missing, incomplete, or only tangentially related, the model *may* supplement with
  general knowledge, tagging those claims with `[G]`.
- On conflict between CONTEXT and general knowledge, say so explicitly and prefer CONTEXT.
- Never fabricate specifics (numbers, dosages, names, dates) under a `[C]` tag that aren't
  actually in the CONTEXT.
- End every answer with a one-line `Grounding: <fully context-based | partially context-based |
  general-knowledge-based>` summary.

`run_ragas_evaluation()` now also scores `answer_correctness` (import: `from ragas.metrics import
answer_correctness`, confirmed working as-is against the pinned `ragas==0.3.9`; no alternate
import path or renamed metric class was needed) alongside `faithfulness` and `answer_relevancy`,
added at the same point in the metrics list where `faithfulness` is conditionally appended (i.e.
whenever `any_answer_generated` is true and an embeddings backend is available — `answer_correctness`
needs both an LLM judge and embeddings, same as `answer_relevancy`). It uses the same
`SingleTurnSample` fields already being populated (`user_input`, `response`, `reference` — the
first `ground_truths` entry), so no new sample construction was needed.

**IMPORTANT — do not "fix" a lower faithfulness score back to the old strict-refusal prompt.**
`faithfulness` measures what fraction of the answer is attributable to the retrieved context; a
blended answer that legitimately draws on general knowledge for gaps in thin context will score
lower on `faithfulness` *by design*. That is expected under this policy, not a regression. Judge
generation quality by looking at `faithfulness` **alongside** `answer_correctness` and
`answer_relevancy`, not `faithfulness` in isolation — a drop in `faithfulness` paired with healthy
`answer_correctness`/`answer_relevancy` means the blending policy is working as intended.
`generate_closed_book_answer()` (the separate no-context function from the prior fix) was **not**
touched — it already invites the model to answer from its own knowledge and was never subject to
the strict-refusal problem.

**Verified live:** ingested a small multi-paragraph test document into a throwaway Qdrant
collection where only part of the content matched the test questions, then ran `--generate`
against questions with deliberately thin/tangential retrieved context. The model answered instead
of refusing, produced answers containing both `[C]` and `[G]` tags, and ended each answer with the
required `Grounding: ...` line. A full `run_ragas_evaluation()` pass with `--generate` printed
`answer_correctness` in the metrics table alongside `faithfulness`/`answer_relevancy` without
errors. Test collection was deleted afterward.

**If you see this again:** check `generate_answer()`'s prompt template in
`scripts/evaluate_ragas.py` for the `[C]`/`[G]` attribution-tagged blending instructions (not the
old "ONLY the provided context" strict-refusal wording), and check that `answer_correctness` is
still imported from `ragas.metrics` and appended alongside `faithfulness`/`answer_relevancy` in
`run_ragas_evaluation()`. A sudden jump in refusal-shaped answers ("I cannot answer this from the
provided context...") is the signature of this bug recurring.

## 2026-07-16 — RAGAS judge LLM truncation, and parent-chunk retrieval was never implemented

Two more bugs found in the same pass as the ingest bugs below.

### Bug 3: RAGAS judge LLM hit `max_tokens` truncation constantly

**Symptom:** Live `ragas.evaluate()` runs repeatedly logged `LLMDidNotFinishException` /
"LLM returned 1 generations instead of requested 3" warnings, producing scattered `NaN` values
across every metric in every run.

**Root cause:** `_default_judge_llm()` in `scripts/evaluate_ragas.py` constructed both
`ChatAnthropic` (direct API path) and `ChatAnthropicVertex` (Vertex fallback path) with no
explicit token-limit parameter, so both silently used the `langchain-anthropic` /
`langchain-google-vertexai` wrapper default of **1024 tokens** — far too short for RAGAS's
multi-generation self-consistency sampling (`n=3`) on anything but the shortest contexts.

**Fix:** Both constructors now pass `max_tokens=4096` explicitly. Verified via
`inspect.signature`/`model_fields` that `ChatAnthropic`'s field is `max_tokens` (alias
`max_tokens_to_sample`) and `ChatAnthropicVertex`'s field is `max_output_tokens` (alias
`max_tokens`) — both accept the `max_tokens=` kwarg through their alias, so the same call shape
works for both. Also bumped `generate_answer()`'s raw Anthropic SDK call (`--generate` path) from
`max_tokens=1024` to `2048`, since answer generation over retrieved context can run long too.

**If you see this again:** check `_default_judge_llm()` in `scripts/evaluate_ragas.py` for an
explicit `max_tokens` on both the `ChatAnthropic(...)` and `ChatAnthropicVertex(...)` calls. Don't
assume the parameter name — verify via `inspect` against the installed `langchain-anthropic` /
`langchain-google-vertexai` versions, since it differs between the two wrappers (and has changed
across versions historically).

### Bug 4: Parent-chunk context expansion was completely unimplemented

**Symptom:** `config/config.yaml` shipped with `retrieval.fetch_parents: true`, and
README.md/SKILL.md describe "hierarchical parent-child retrieval" — small chunks for precise
search, larger parent chunks retrieved for full context — as a core feature. In reality, no code
ever read `fetch_parents`, and parent chunk text was never stored anywhere retrievable at all.

**Root cause:** `EfficientRAG.ingest()` only ever embedded and upserted `child`-type chunks
(`if chunk.chunk_type != "child": continue`); parent chunk text was discarded after chunking.
Child payloads stored a `parent_id` string, but there was no corresponding parent record for
anyone to look up. `EfficientRAG.retrieve()` never referenced `config["retrieval"]["fetch_parents"]`
at all — it always returned the terse child-chunk text as `"text"`.

**Fix:**
- `ingest()` now also stores each parent chunk as a separate Qdrant point in the same collection,
  with a **zero dummy `"dense"` vector** (never meant to be searched, only looked up) and
  `chunk_type: "parent"` in the payload. The point ID is a **deterministic UUID5** derived from the
  chunk's `parent_id` string (`EfficientRAG._parent_point_id()`), so a parent chunk can be fetched
  directly via `qdrant.retrieve(ids=[...])` instead of a filtered scroll/search.
- `retrieve()`'s dense/sparse search queries now explicitly filter on `chunk_type: "child"` —
  without this, the zero-vector parent points were showing up as search hits themselves (and, via
  the cross-encoder rerank stage, sometimes ranking well since parent text often contains the
  matching content), which is not what "search on small chunks" is supposed to mean.
- When `retrieval.fetch_parents` is true, `retrieve()` now looks up each hit's parent chunk by its
  deterministic point ID and expands the returned `"text"` field to the full parent text, while
  keeping the original child chunk text available under a new `"child_text"` field. Lookup failures
  (e.g. points ingested before this fix, with no corresponding parent point) fall back gracefully
  to child-only text — never raises. When `fetch_parents` is false, behavior is unchanged from
  before (child text only).
- `scripts/retrieve.py`'s `format_context()` now uses the (possibly parent-expanded) `"text"` field
  for the main "Context:" block and shows `"child_text"` as a separate "Matched excerpt:" line when
  it differs from `"text"` — old result dicts without `"child_text"` are handled gracefully (the
  field is just absent, no crash).

**Verified live:** ingested a small multi-paragraph test doc (2 parent chunks, 10 child chunks) —
`ingest()` reported `child_points_ingested: 10, parent_points_ingested: 2` and Qdrant's collection
`points_count` was 12. A `chunk_type: parent` scroll filter returned both parent points with full
text. A `retrieve()` call with `fetch_parents: true` returned only `chunk_type: "child"` hits (the
new search filter working) with `"text"` expanded to the full ~1700-char parent chunk and
`"child_text"` holding the original ~40-400 char matched snippet. The same query against
`fetch_parents: false` returned `"text" == "child_text"` (short excerpts only), confirming the
old behavior is preserved when the feature is off. Test collection was deleted afterward.

**If you see this again:** check `EfficientRAG._parent_point_id()`, the parent-point branch in
`ingest()`, and the `chunk_type: "child"` filter + parent-expansion block in `retrieve()`, all in
`scripts/utils.py`.

## 2026-07-16 — Two multi-document ingest bugs found and fixed

Both bugs were **invisible on single-file smoke tests** and only surfaced once ingestion was run
against a real multi-document corpus (120 documents, NFCorpus-derived). They were found together
because fixing the first one immediately exposed the second.

### Bug 1: `--recursive` directory ingest silently wiped all previously-ingested files

**Symptom:** After a `--recursive` ingest of a directory with many files, the collection ended up
with only a handful of points — roughly what a *single* document would produce, not the whole
corpus. At query time, this looked exactly like a ranking or embedding bug: **retrieval returned
the same generic, seemingly irrelevant chunk for almost every query**, regardless of how the query
was phrased, because the collection actually only contained chunks from the *last* ingested file
(or, more precisely, whichever file's `force_recreate=True` call ran last before final upsert
ordering settled).

**Root cause:** `scripts/ingest.py`'s directory-ingest loop passed
`force_recreate=args.force_recreate` on **every** file in the loop, not just the first one. Since
`EfficientRAG.create_collection()` treated `force_recreate=True` as "delete then recreate the
collection," every single file's `ingest()` call — including the 2nd, 3rd, ..., 120th — deleted
the collection and recreated it from scratch, discarding every previously-ingested file's chunks
in the process. Only the most recently ingested file's points ever survived to the end of the run.

**Why it was hard to catch:** A single-file ingest (`--path report.pdf`, no `--recursive`) only
ever calls `create_collection`/`ingest` once, so `force_recreate` behaves exactly as expected and
the bug never triggers. It only appears once you ingest more than one file in the same
`--recursive` run — and even then, the ingestion CLI's own per-file print output (`Ingesting file:
...`, then a per-file stats dict) looks completely normal; nothing errors, nothing warns. The only
externally visible symptom is at **query time**, on a fully-populated-looking collection that is
actually mostly empty of everything except the last file.

**Fix:** `scripts/ingest.py`'s directory-ingest loop now only sets `force_recreate=True` for the
*first* file processed; every subsequent file in the same loop passes `force_recreate=False`
(reusing the collection instead of recreating it):

```python
recreate_pending = args.force_recreate
for f in files:
    ...
    stat = rag.ingest(str(f), args.collection, tags=tags, force_recreate=recreate_pending)
    recreate_pending = False
    stats_list.append(stat)
```

**If you see this again:** check `scripts/ingest.py`'s directory-ingest loop first. Confirm
`force_recreate` is only ever `True` for the first file processed in a given run, and confirm the
collection's `points_count` (via `get_stats()` / `scripts/retrieve.py`) roughly matches "sum of
chunks across all ingested files," not "chunks from one file."

### Bug 2: 409 Conflict on the second file onward, after fixing Bug 1

**Symptom:** Once Bug 1 was fixed (so `force_recreate` was `False` for every file after the
first), ingesting the second and subsequent files in a directory started failing with a
**409 Conflict** from Qdrant.

**Root cause:** `EfficientRAG.create_collection()` unconditionally called
`self.qdrant.create_collection(...)` regardless of whether the collection already existed. With
`force_recreate=False` and the collection already present (created by the first file), Qdrant's
`create_collection` API call correctly rejected the duplicate-creation attempt with a 409.

**Fix:** `create_collection()` now checks `self.qdrant.collection_exists(full_name)` first. If the
collection already exists and `force_recreate` is not set (or was already handled), it logs
`"Collection already exists: ... — reusing"` and returns early without calling
`create_collection()` again. It only calls Qdrant's `create_collection()` when the collection does
not yet exist (either because it never did, or because `force_recreate=True` just deleted it).

**If you see this again:** check `EfficientRAG.create_collection()` in `scripts/utils.py`. Confirm
it calls `qdrant.collection_exists()` before ever calling `qdrant.create_collection()`, and that a
collection which already exists (with `force_recreate=False`) short-circuits to a no-op instead of
attempting to recreate it.

### Combined takeaway

These two bugs are why **multi-document ingestion behavior should always be spot-checked with
`get_stats()` after a `--recursive` run**, not just a "did it print errors" check. A `--recursive`
ingest of N files should produce roughly N documents' worth of points in the collection; if the
final `points_count` looks like it only reflects one document, suspect `force_recreate` handling
in the ingest loop before suspecting the embedding model, chunking, or retrieval ranking logic.

## In progress — hyperparameter tuning pass (results pending)

A hyperparameter tuning pass is currently running against a 120-document real corpus
(NFCorpus-derived), now that the two bugs above are fixed and no longer silently corrupting
ingestion at scale. It covers:

- A grid over `indexing.child_chunk_size` (and correspondingly `indexing.parent_chunk_size`) ×
  `retrieval.rerank_candidate_pool`.
- A sweep over `retrieval.rrf_k` (the client-side RRF rank-fusion smoothing constant).
- A baseline-vs-tuned RAGAS validation run (`context_precision`, `context_recall`, `faithfulness`,
  `answer_relevancy`) comparing the current defaults against the best configuration found.

**Results are not available yet.** Do not cite specific RAGAS scores, recall numbers, or "optimal"
hyperparameter values for this project until this run completes and this section is updated. The
current defaults in `config/config.yaml` (`child_chunk_size: 400`, `rerank_candidate_pool: 40`,
`rrf_k: 60`) are reasonable starting points taken from general Qdrant/reranking guidance (see
`docs/ARCHITECTURE.md` Section 3), not yet validated as optimal for this specific corpus.

## Design change — RRF fusion moved from server-side to client-side

The original architecture plan (see `docs/ARCHITECTURE.md` history) called for fusing dense +
sparse hybrid search results using Qdrant's server-side `FusionQuery(fusion=Fusion.RRF)` inside a
single `query_points()` call with two `Prefetch` legs. During implementation this was changed:
`EfficientRAG.retrieve()` now issues two separate `query_points()` calls (one per named vector,
`"dense"` and `"sparse"`) and fuses the results itself using the standard RRF formula
(`score += 1 / (rrf_k + rank)` per leg, summed across legs). This was a deliberate choice, not a
bug: Qdrant's server-side RRF fusion does not expose a tunable `k` constant (confirmed via
[Qdrant GitHub issue #5116](https://github.com/qdrant/qdrant/issues/5116)), and this project wanted
`k` to be tunable as part of the hyperparameter search described above. The tunable constant is
exposed as `retrieval.rrf_k` in `config/config.yaml` (default `60`).

## 2026-07-16 — NeuralWatt provider scaffolding added (`_neuralwatt_llm()`, NOT live-verified, no calls made)

**What this adds:** Scaffolding in `scripts/evaluate_ragas.py` for a NEW candidate judge/generation
provider, NeuralWatt — a hosted inference platform believed, but **not yet confirmed**, to expose
an OpenAI-API-compatible endpoint:

- `NEURALWATT_MODELS`: a dict documenting three model IDs this project has looked at for possible
  use — `glm-5.2-short` (reasoning, 200K context, $1.45/$4.50 per M input/output tokens, ~1.59
  Wh/request — chosen over the full `glm-5.2` for generation+judging because it keeps the same
  "Reasoning" capability tag at a near-identical price with lower energy, and 200K context already
  exceeds this project's actual chunk/context sizes), `kimi-k2.7-code` (reasoning + native JSON
  mode, for judging — native JSON mode may resolve the RAGAS schema-validation NaN issue more
  reliably than Ollama's generic `format="json"` flag), and `qwen3.5-397b` (reasoning + JSON mode,
  a backup/third multi-judge consensus member, replacing `minimax-m3` since NeuralWatt doesn't
  offer it). These figures were transcribed from a pasted NeuralWatt pricing table, not confirmed
  against a live API call.
- `_neuralwatt_llm(model_name, temperature=0.1)`: builds a `langchain_openai.ChatOpenAI` wrapped in
  `LangchainLLMWrapper`, following the exact same construction pattern as `_ollama_judge_llm()`.
  Reads `NEURALWATT_API_KEY` from the environment and raises a `RuntimeError` with an actionable
  message if it isn't set (no silent fallback to a fake key or another provider, unlike
  `_default_judge_llm()`/`_default_judge_embeddings()`'s cross-provider fallback elsewhere in this
  module). Constructs the client only — never calls `.invoke()`.
- `--neuralwatt-judge-models` CLI flag: a documented-but-inert extension point showing how
  NeuralWatt models would eventually be added as extra `--multi-judge-crosscheck` members via
  `_neuralwatt_llm()`. Passing it today only prints what *would* happen and returns immediately —
  it does not run an evaluation, does not change `run_multi_judge_faithfulness_crosscheck()`'s
  default behavior, and makes no network call. That function's docstring gained a "FUTURE
  NEURALWATT WIRING" section spelling out the (unimplemented) integration shape.

**IMPORTANT — base URL is UNCONFIRMED, no live calls have been tested:** `_neuralwatt_llm()`
defaults `base_url` to `https://api.neuralwatt.com/v1` (overridable via `NEURALWATT_BASE_URL`).
This is a placeholder guess, not a value taken from verified NeuralWatt documentation or a
successful request — NeuralWatt's OpenAI-API-compatibility itself is also unconfirmed. Verify both
before ever invoking this function for real. This change was implementation-only: no network call
was made to `neuralwatt.com`, `api.neuralwatt.com`, or any other NeuralWatt domain while adding
this scaffolding, and no real or fake `NEURALWATT_API_KEY` was set or used.

`langchain_openai` was already a pinned dependency (`requirements.txt`, used lazily elsewhere for
`OpenAIEmbeddings` in `_default_judge_embeddings()`) — `_neuralwatt_llm()` imports `ChatOpenAI`
from it lazily inside the function, matching that existing lazy-import convention (and
`_build_anthropic_client()`'s `try`/`except ImportError` style) rather than adding a new top-level
import.
