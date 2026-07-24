# FAQ Pipeline Debug — Quick Start

## TL;DR

The FAQ pipeline has **3 stages**:

1. **Analyze** → Identify gaps in `kb_coverage`
2. **Generate** → Create FAQ articles in `generated_articles`
3. **Publish** → Push to Confluence, index to `kb_articles`

If Access/Permissions FAQs aren't appearing, one stage is blocked.

---

## Run Diagnostics (OCP)

```bash
# Get pod name
POD=$(oc get pods -n jira-messaging--runtime-ext -l app=ai-helpdesk-agent -o jsonpath='{.items[0].metadata.name}')

# Run Python diagnostic script
oc exec -n jira-messaging--runtime-ext $POD -- python3 debug_faq_pipeline.py
```

Output will show:
- ✓ Recent FAQ articles
- ✓ Resolved tickets by category (total, resolved, with_summary)
- ✓ KB coverage for Access/Permissions themes
- ✓ Ungenerated gaps (missing FAQ articles)
- ✓ Root cause diagnosis

---

## Or Query Database Directly

```bash
# Connect to Postgres
psql $DATABASE_URL < faq_pipeline_diagnostics.sql
```

This runs 9 diagnostic queries:
1. FAQ article status (draft vs published)
2. Resolved tickets by category
3. KB coverage themes
4. Ungenerated gaps
5. Analysis freshness
6. Generation status
7. Publishing status
8. Sample source tickets
9. Root cause analysis

---

## Most Likely Issues (In Order)

### 1. Gap Analysis Never Ran
**Check**: 
```sql
SELECT COUNT(*) FROM kb_coverage
WHERE theme ILIKE '%access%' OR theme ILIKE '%permiss%';
```
**Expected**: > 5  
**If 0**: Run `python faq_service.py analyze`

### 2. No Resolved Tickets with `resolution_summary`
**Check**:
```sql
SELECT COUNT(*) FROM tickets t
JOIN ticket_classifications tc ON t.ticket_key = tc.ticket_key
WHERE tc.category IN ('Access','Permissions')
  AND t.resolution IS NOT NULL
  AND t.resolution_summary IS NOT NULL;
```
**Expected**: > 5  
**If 0**: Ingest is not populating `resolution_summary`

### 3. Generator Skipping Gaps (Silent Fail)
**Check**:
```sql
SELECT theme FROM kb_coverage
WHERE coverage_status IN ('missing','partial')
  AND theme NOT IN (SELECT article_topic FROM generated_articles WHERE format='faq');
```
**If > 0**: Generator is being skipped due to:
- No source tickets found → generator.py line 147
- Semantic duplicate detected → generator.py line 319
- Gemini generation error → generator.py line 209

**Fix**: Check logs for "Failed to generate FAQ" or "Skipping...duplicate"

### 4. Articles Generated but Not Published
**Check**:
```sql
SELECT COUNT(*) FROM generated_articles
WHERE format = 'faq' AND status = 'draft';
```
**If > 0**: Run `python faq_service.py publish`

### 5. Published but Not Indexed
**Check**:
```sql
SELECT COUNT(*) FROM kb_articles
WHERE topics_covered ILIKE '%access%' OR topics_covered ILIKE '%permiss%';
```
**If 0**: `kb_articles` indexing job hasn't run after publishing

---

## Manual Debug (Single Theme)

If diagnosis shows gaps aren't being generated, test manually:

```bash
POD=$(oc get pods -n jira-messaging--runtime-ext -l app=ai-helpdesk-agent -o jsonpath='{.items[0].metadata.name}')

# Generate for a specific theme to see detailed errors
oc exec -n jira-messaging--runtime-ext $POD -- python3 -m faq.generator \
    --theme "Your Access/Permissions Theme Name" \
    --debug
```

This will show:
- Source tickets found (or why not)
- Gemini response (or error)
- Dedup check results

---

## Log Inspection

Check pod logs for generation failures:

```bash
oc logs -n jira-messaging--runtime-ext $POD --tail=500 | grep -E 'FAQ|generate|access|permission'
```

Look for:
- `Failed to generate FAQ` → Gemini API issue
- `No source tickets` → Ticket matching issue
- `Skipping.*duplicate` → Dedup threshold too aggressive
- `401|403` → Auth issue

---

## Pipeline Execution

Full pipeline: `analyze` → `generate` → `export` → `publish`

```bash
# Full run
python faq_service.py run

# Or step-by-step
python faq_service.py analyze      # Identify gaps
python faq_service.py generate     # Create FAQs
python faq_service.py export       # Write to output Google Doc
python faq_service.py publish      # Push to Confluence
```

---

## Files

| File | Purpose |
|------|---------|
| `debug_faq_pipeline.py` | Python diagnostic script (run on OCP) |
| `faq_pipeline_diagnostics.sql` | SQL diagnostics (run on DB) |
| `FAQ_PIPELINE_DEBUG_REPORT.md` | Full technical explanation |
| `faq_service.py` | CLI entry point (lines 199-375 for pipeline) |
| `faq/analyzer.py` | Gap analysis (lines 69-189) |
| `faq/generator.py` | FAQ generation (lines 213-358) |
| `generation/publisher.py` | Confluence publishing (lines 49-109) |

---

## Config Check

Verify these env vars are set:

```bash
# Gap analysis
echo $CONFLUENCE_KB_SPACE          # Should be 'HUB' or similar
echo $FAQ_CONFLUENCE_SPACES        # Should be 'HUB,OMEGA' or similar

# Generation
echo $DEDUP_COSINE_THRESHOLD       # Default 0.75
echo $GEMINI_MODEL_GENERATION      # Should be gemini-3.1-pro-preview

# Publishing
echo $CLOUD_URL                    # Should be https://...atlassian.net
echo $CONFLUENCE_CLIENT_ID         # Should be set
```

---

## Still Stuck?

1. Run `debug_faq_pipeline.py` — identifies which stage is blocked
2. Check the "Root Cause Analysis" section in the output
3. Review logs for specific error messages
4. Query `kb_coverage` to see what gaps were identified
5. Query `generated_articles` to see what was generated vs what's missing
6. Check if Gemini API quotas are exceeded
7. Verify Confluence space exists and user has write permissions

---

## Contact / Escalate

If diagnostics show:
- **Gemini timeouts** → Increase `GEMINI_*_TIMEOUT` in config
- **Confluence auth failures** → Verify OAuth credentials and space permissions
- **Database locked** → Check for concurrent pipeline runs
- **Duplicate threshold too aggressive** → Lower `DEDUP_COSINE_THRESHOLD`

See `FAQ_PIPELINE_DEBUG_REPORT.md` for detailed explanations.
