<!-- Mirrored from Confluence: https://<YOUR_DOMAIN>.atlassian.net/wiki/spaces/~ceverson/pages/395772365 -->
<!-- Last synced: 2026-05-29 -->

# Auto-Responder

> **Scope:** This page covers agent-facing draft behavior and the drafting pipeline. It does not cover routing decisions (see Architecture → M5 Router) or model training (see Classifier & ML Pipeline).

## What It Does

When a ticket arrives in <PROJECT_KEY>, the auto-responder reads it, searches the knowledge base, and posts a draft response as an **internal comment** — visible only to agents, not to the customer.

Drafts improve over time as agents rate them. The more agents interact with the system, the better future drafts become for similar tickets.

---

## For Agents: Using the Auto-Responder

### What a Draft Looks Like

Drafts appear as internal comments on the ticket labeled by type:

**Self-Service** — the customer can resolve this themselves (links to docs, step-by-step)

**Admin Action** — an admin needs to do something first (includes technician steps + customer response)

**Needs Info** — not enough detail to act (asks specific questions)

### Rating Drafts

Post a single emoji as an **internal comment** to rate the draft:

| Emoji | What you're saying |
|---|---|
| ✅ | Both the customer response and technician steps are good |
| 👤 | Customer response is good, technician steps need work |
| 🔧 | Technician steps are good, customer response needs work |
| ❌ | Both are off — wrong type or wrong content |
| 🔄 | Response type is wrong (e.g., drafted as self-service but actually admin action) |
| ❓ | Should have been "Needs Info" |

### Requesting a New Draft

Post `/ai-lookup` as an **internal comment**. The service re-reads the full thread (including latest customer reply) and posts an updated draft.

### Requesting a Disposition Review

Post `/ai-review` as an **internal comment** for a recommendation:

| Disposition | What it means |
|---|---|
| `close` | Issue is resolved — includes a draft closing message |
| `sprint_work` | Requires engineering work beyond helpdesk scope |
| `needs_action` | Customer replied with new information — follow-up needed |
| `stale` | No customer response after your last message |

Post `🤖` for both a new draft and disposition review at once.

---

## For Developers: How It Works

### When Drafts Are Posted

1. **On ticket creation** — webhook triggers `handle_new_ticket()` immediately
2. **On every pipeline cycle** — the auto-draft sweep catches missed tickets

Before drafting: assignee check (unless `AUTO_DRAFT_ALL=true`), existing response check, content match check. If no matching FAQ/KB content is found, drafting is skipped.

### The Draft Pipeline

```
lookup(ticket summary)
  ├─ FAQ entries in database (category/issue_type match)
  ├─ Confluence KB articles (CQL keyword search)
  ├─ Resolved tickets with similar classification + resolution summary
  └─ Indexed Atlassian support/developer doc URLs

draft_response(ticket summary, description, matched content)
  ├─ Build context block from matches (up to 4,000 chars)
  ├─ Add up to 5 semantically similar few-shot examples from the ANN index
  └─ Gemini: classify type + write customer response + write technician steps

post_internal_comment(ticket key, draft)
  └─ POST /servicedeskapi/request/{key}/comment {"body": "...", "public": false}
```

### How the Model Improves Over Time

Every draft is recorded. After an agent responds, their actual response is scored using **embedding cosine similarity** (all-MiniLM-L6-v2, 384-dim vectors) per ANTSE-295. Approved draft pairs and harvested organic responses are embedded into the responder ANN index.

The ANN index is rebuilt nightly from approved feedback rows and resolved-ticket examples, then queried at draft time. A similarity floor keeps weak matches out of the prompt; if the index is empty, the system falls back to the legacy static top-5 selection.

The model doesn't need retraining — it gets better as your team uses it.

### Measuring ANN Few-Shot Impact (ANTSE-310)

Compare draft acceptance **one week before** enabling ANN retrieval against **one week after**. Run these queries against the production database (or a recent backup):

**Baseline window (7 days before deploy):**
```sql
SELECT
    feedback_category,
    COUNT(*) AS draft_count,
    ROUND(AVG(similarity_score), 3) AS avg_similarity
FROM ai_draft_feedback
WHERE captured_at >= datetime('now', '-14 days')
  AND captured_at < datetime('now', '-7 days')
  AND feedback_category IS NOT NULL
GROUP BY feedback_category
ORDER BY draft_count DESC;
```

**Post-deploy window (7 days after deploy):**
```sql
SELECT
    feedback_category,
    COUNT(*) AS draft_count,
    ROUND(AVG(similarity_score), 3) AS avg_similarity
FROM ai_draft_feedback
WHERE captured_at >= datetime('now', '-7 days')
  AND feedback_category IS NOT NULL
GROUP BY feedback_category
ORDER BY draft_count DESC;
```

**Acceptance rate (agent-used drafts):**
```sql
-- Replace date bounds with your before/after windows
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN feedback_category IN ('as_is', 'lightly_edited') THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0),
        1
    ) AS acceptance_pct,
    COUNT(*) AS scored_drafts
FROM ai_draft_feedback
WHERE captured_at >= datetime('now', '-7 days')
  AND feedback_category IS NOT NULL
  AND feedback_category != 'ignored';
```

Record baseline and post-deploy `acceptance_pct` values in the deploy notes. Target: stable or improved acceptance with no increase in `heavily_rewritten` / `ignored` share.
