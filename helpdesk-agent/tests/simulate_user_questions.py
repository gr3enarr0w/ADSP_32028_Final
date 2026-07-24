#!/usr/bin/env python3
"""Simulate fake support tickets through the full lookup → draft response pipeline.

Seeds a temp DB with realistic data, submits 10 fake tickets (summary + description
like a real user would write), runs lookup, and drafts a technician-style response
with inline hyperlinks. Output goes to tests/simulation_results.md.

Usage:
    python tests/simulate_user_questions.py
"""

import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results.md")


# ── Seed data ────────────────────────────────────────────────────────────────

ATLASSIAN_DOCS = [
    ("https://support.atlassian.com/jira-service-management-cloud/docs/configure-sla-policies/",
     "jira-service-management-cloud", "Configure Sla Policies"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/set-up-queues/",
     "jira-service-management-cloud", "Set Up Queues"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/manage-request-types/",
     "jira-service-management-cloud", "Manage Request Types"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/configure-customer-notifications/",
     "jira-service-management-cloud", "Configure Customer Notifications"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/set-up-your-service-project/",
     "jira-service-management-cloud", "Set Up Your Service Project"),
    ("https://support.atlassian.com/jira-cloud/docs/configure-workflows/",
     "jira-cloud", "Configure Workflows"),
    ("https://support.atlassian.com/jira-cloud/docs/manage-project-permissions/",
     "jira-cloud", "Manage Project Permissions"),
    ("https://support.atlassian.com/jira-cloud/docs/configure-custom-fields/",
     "jira-cloud", "Configure Custom Fields"),
    ("https://support.atlassian.com/jira-cloud/docs/bulk-change-issues/",
     "jira-cloud", "Bulk Change Issues"),
    ("https://support.atlassian.com/confluence-cloud/docs/create-and-edit-pages/",
     "confluence-cloud", "Create And Edit Pages"),
    ("https://support.atlassian.com/confluence-cloud/docs/configure-space-permissions/",
     "confluence-cloud", "Configure Space Permissions"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/use-automation-rules/",
     "jira-service-management-cloud", "Use Automation Rules"),
    ("https://support.atlassian.com/jira-cloud/docs/migrate-from-server-to-cloud/",
     "jira-cloud", "Migrate From Server To Cloud"),
    ("https://support.atlassian.com/jira-service-management-cloud/docs/manage-agents-and-customers/",
     "jira-service-management-cloud", "Manage Agents And Customers"),
    ("https://support.atlassian.com/jira-cloud/docs/configure-dashboards/",
     "jira-cloud", "Configure Dashboards"),
]

FAQ_ENTRIES = [
    ("SLA configuration after migration",
     "How to reconfigure SLAs after DC-to-Cloud migration",
     "<p>After migrating to Cloud, SLA policies need to be reconfigured because "
     "Cloud uses a different SLA engine. Steps: 1) Navigate to Project Settings > SLAs. "
     "2) Review migrated SLA definitions. 3) Update calendar hours. 4) Test with a sample ticket.</p>",
     "https://wiki.example.com/display/FAQ/SLA+Reconfiguration"),
    ("Permission scheme differences Cloud vs DC",
     "Understanding permission changes in Cloud",
     "<p>Cloud uses a simplified permission model compared to Data Center. Key differences: "
     "1) Global permissions are managed in admin.atlassian.com. 2) Project roles map differently. "
     "3) Some custom permission types are not available in Cloud.</p>",
     "https://wiki.example.com/display/FAQ/Permission+Changes"),
    ("Custom field migration issues",
     "Resolving custom field problems after migration",
     "<p>Custom fields may need attention after migration. Common issues: "
     "1) Select list options may have changed order. 2) Calculated fields need reconfiguration. "
     "3) Cascading selects require manual validation.</p>",
     "https://wiki.example.com/display/FAQ/Custom+Field+Issues"),
    ("Automation rules not firing",
     "Troubleshooting automation rules after Cloud migration",
     "<p>Automation rules from DC may not transfer cleanly to Cloud. Check: "
     "1) Rule is enabled. 2) Triggers reference correct events. "
     "3) Conditions use Cloud-compatible JQL. 4) Actions have proper permissions.</p>",
     "https://wiki.example.com/display/FAQ/Automation+Troubleshooting"),
]

KB_ARTICLES = [
    ("KB-001", "MIGRATE", "Migration Checklist for Jira DC to Cloud",
     "Complete checklist covering pre-migration assessment, data backup, "
     "user mapping, configuration review, test migration, and go-live steps.",
     "https://wiki.example.com/display/MIGRATE/Migration+Checklist",
     "migration,checklist,planning", "migration,planning,assessment"),
    ("KB-002", "MIGRATE", "Known Issues After Cloud Migration",
     "List of known issues encountered after migrating from DC to Cloud including "
     "dashboard gadget incompatibilities, workflow validator changes, and email handler differences.",
     "https://wiki.example.com/display/MIGRATE/Known+Issues",
     "known-issues,post-migration", "known-issues,dashboards,workflows,email"),
    ("KB-003", "SUPPORT", "How to Request Access to Cloud Applications",
     "Step-by-step guide for requesting access to Jira Cloud, Confluence Cloud, "
     "and JSM Cloud through the IT service desk portal.",
     "https://wiki.example.com/display/SUPPORT/Request+Access",
     "access,onboarding", "access,permissions,onboarding"),
]

RESOLVED_TICKETS = [
    ("<PROJECT_KEY>-1001", "SLA not working after migration",
     "SLA timers show as paused for all tickets in the service project",
     "Done", "Done", "Configuration",
     "SLA configuration", "sla migration timer paused",
     "Reconfigured SLA calendar to use Cloud business hours. The DC calendar "
     "format was not compatible. Reset SLA on affected tickets via bulk change."),
    ("<PROJECT_KEY>-1002", "Cannot assign tickets to group",
     "After migration, the team group is not appearing in the assignee dropdown",
     "Done", "Done", "Access",
     "Group membership", "group assignee permissions",
     "Group was migrated but not added to the project role. Added the group "
     "to the 'Service Desk Team' project role in Cloud project settings."),
    ("<PROJECT_KEY>-1003", "Dashboard gadgets missing after migration",
     "Several custom dashboard gadgets from DC are not showing in Cloud",
     "Done", "Done", "UI/UX",
     "Dashboard migration", "dashboard gadgets missing widgets",
     "Some DC gadgets have no Cloud equivalent. Replaced with Cloud-native "
     "gadgets: Pie Chart, Filter Results, and Two-Dimensional Filter."),
    ("<PROJECT_KEY>-1004", "Workflow transition screen not showing custom fields",
     "Custom fields on the workflow transition screen are blank after migration",
     "Done", "Done", "Configuration",
     "Workflow screens", "workflow transition screen custom fields",
     "Transition screens needed to be re-associated in Cloud. The screen "
     "scheme mapping was not preserved during migration. Reconfigured in "
     "Project Settings > Screens."),
    ("<PROJECT_KEY>-1005", "Customer notifications not being sent",
     "Customers report they are not receiving email notifications from JSM Cloud",
     "Done", "Done", "Notifications",
     "Email notifications", "notifications email customer jsm",
     "Cloud uses a different notification scheme. Enabled customer notifications "
     "in Project Settings > Customer notifications. Also verified the customer's "
     "email was linked to their Atlassian account."),
    ("<PROJECT_KEY>-1006", "Automation rule runs but does not update field",
     "An automation rule triggers on ticket creation but the field update action fails silently",
     "Done", "Done", "Configuration",
     "Automation rules", "automation rule field update silent failure",
     "The automation rule was using a DC-only smart value syntax. Updated to "
     "Cloud-compatible syntax: {{issue.fields.customfield_10001}} to {{issue.customfield_10001}}."),
]


# ── Fake tickets (like real users would submit) ──────────────────────────────

FAKE_TICKETS = [
    {
        "key": "<PROJECT_KEY>-2001",
        "summary": "SLA timers all say paused since we moved to cloud",
        "description": (
            "Hi, since we migrated to Jira Cloud last week all of our SLA timers "
            "are showing as 'Paused' even on active tickets. We have 3 SLA policies "
            "set up — Time to First Response, Time to Resolution, and Ongoing. "
            "None of them are counting down. Our team is getting dinged on SLA "
            "compliance and we need this fixed ASAP."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2002",
        "summary": "Dashboard broken - all gadgets show error",
        "description": (
            "My main dashboard that I use every day is completely broken after the "
            "migration. Every gadget says 'Unable to load gadget' or just shows a "
            "grey box. I had a pie chart for issue types, a filter results gadget "
            "for my queue, and a two-dimensional filter. Can you help me get these back?"
        ),
    },
    {
        "key": "<PROJECT_KEY>-2003",
        "summary": "Can't add our dev team group as assignee",
        "description": (
            "We're trying to assign tickets to our development team group but the "
            "group doesn't show up in the assignee field. The group name is "
            "'platform-engineering' and it existed in DC. I can see it in "
            "admin.atlassian.com under groups but it doesn't appear in the project."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2004",
        "summary": "automation rule to set priority isn't working anymore",
        "description": (
            "We had an automation rule in DC that set ticket priority to Critical "
            "when the customer type is VIP. After migration the rule shows as "
            "enabled but when a VIP submits a ticket the priority stays at Medium. "
            "I checked the rule log and it says 'action failed' but no details."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2005",
        "summary": "Custom fields missing from ticket create screen",
        "description": (
            "When I create a new ticket in our migrated Cloud project, several "
            "custom fields are gone. Specifically 'Environment', 'Business Unit', "
            "and 'Impact Assessment' which are required fields for our process. "
            "The fields exist in the system but aren't on the create screen."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2006",
        "summary": "Customers not receiving any emails from service desk",
        "description": (
            "Multiple customers have reported they aren't getting notification "
            "emails when we comment on their tickets or when status changes. "
            "This started right after the cloud migration. I checked their "
            "notification preferences and everything looks enabled."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2007",
        "summary": "Who do I contact to get Cloud access for new hires?",
        "description": (
            "We have 5 new team members starting next Monday and they need access "
            "to Jira Cloud and Confluence Cloud. In the old DC system I could just "
            "add them to the LDAP group but I'm not sure how onboarding works now. "
            "What's the process?"
        ),
    },
    {
        "key": "<PROJECT_KEY>-2008",
        "summary": "Need to set up queues for our new cloud service project",
        "description": (
            "We're setting up our service project in Cloud and need to create "
            "queues similar to what we had in DC. We need a queue for each team "
            "(Network, Desktop, Cloud Ops) filtered by component. Can you point "
            "me to docs on how queues work in Cloud?"
        ),
    },
    {
        "key": "<PROJECT_KEY>-2009",
        "summary": "Workflow transition buttons disappeared",
        "description": (
            "After migration, the transition buttons on our tickets look different. "
            "In DC we had custom transition screens that asked for a resolution "
            "comment and a root cause field. Now when I click 'Resolve' it just "
            "resolves immediately without asking for any info. We need those "
            "screens back."
        ),
    },
    {
        "key": "<PROJECT_KEY>-2010",
        "summary": "How to connect Slack alerts to Bitbucket deployment pipeline",
        "description": (
            "I want to set up alerts in our #deployments Slack channel whenever "
            "a Bitbucket pipeline deploys to production. This isn't really a "
            "migration issue, just wondering if the team can help set this up "
            "in our new Cloud environment."
        ),
    },
]


def seed_database():
    """Populate the temp DB with all source types."""
    from db import get_db_conn

    now = datetime.now(timezone.utc).isoformat()
    with get_db_conn() as conn:
        for url, product, title in ATLASSIAN_DOCS:
            conn.execute(
                "INSERT INTO atlassian_docs (url, product, title, fetched_at) VALUES (?,?,?,?)",
                (url, product, title, now),
            )

        for topic, title, body_html, confluence_url in FAQ_ENTRIES:
            conn.execute(
                "INSERT INTO generated_articles (article_topic, title, body_html, format, status, confluence_url) "
                "VALUES (?,?,?,'faq','draft',?)",
                (topic, title, body_html, confluence_url),
            )

        for page_id, space, title, body, url, labels, topics in KB_ARTICLES:
            conn.execute(
                "INSERT INTO kb_articles (page_id, space_key, title, body_text, url, labels, topics_covered, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (page_id, space, title, body, url, labels, topics, now),
            )

        for key, summary, desc, status, resolution, category, issue_type, keywords, res_summary in RESOLVED_TICKETS:
            conn.execute(
                "INSERT INTO tickets (ticket_key, summary, description, status, resolution, source, fetched_at) "
                "VALUES (?,?,?,?,?,'cloud',?)",
                (key, summary, desc, status, resolution, now),
            )
            conn.execute(
                "INSERT INTO ticket_classifications "
                "(ticket_key, category, issue_type, keywords, has_resolution, resolution_summary, confidence, classified_at) "
                "VALUES (?,?,?,?,1,?,0.9,?)",
                (key, category, issue_type, keywords, res_summary, now),
            )


def _pick_best_link(matches: dict, ticket_summary: str) -> tuple[str, str] | None:
    """Pick the single most relevant link from all match sources.

    Returns (title, url) of the best match, or None.
    Prioritizes: resolved ticket with resolution > FAQ with URL > KB > Atlassian doc.
    """
    summary_lower = ticket_summary.lower()

    # 1. Resolved tickets — best source since they have proven resolutions
    for t in matches.get("ticket_matches", []):
        if t.get("resolution_summary"):
            return (t["summary"], f"https://test-instance.atlassian.net/browse/{t['ticket_key']}")

    # 2. FAQ entries with Confluence URLs
    for faq in matches.get("faq_matches", []):
        if faq.get("confluence_url"):
            return (faq["title"], faq["confluence_url"])

    # 3. KB articles
    for kb in matches.get("kb_matches", []):
        if kb.get("url"):
            return (kb["title"], kb["url"])

    # 4. Atlassian docs
    for doc in matches.get("atlassian_matches", []):
        return (doc["title"], doc["url"])

    return None


def draft_technician_response(ticket: dict, matches: dict) -> str:
    """Draft a realistic technician response with inline links.

    This templates what Gemini would produce — a warm, concise response
    that identifies the issue and links to one specific article.
    No Gemini API call needed for the simulation.
    """
    summary = ticket["summary"]

    if not matches["found"]:
        return (
            f"Thanks for reaching out. I've looked into your request regarding "
            f"\"{summary}\" but I wasn't able to find a matching article or "
            f"previous resolution in our knowledge base. I'll investigate further "
            f"and get back to you shortly."
        )

    lines = []

    # Opening — acknowledge the specific issue
    lines.append(
        f"Thanks for reaching out. I've taken a look at your issue "
        f"regarding \"{summary}\" and I have some information that should help."
    )
    lines.append("")

    # Resolved ticket match — most valuable, cite the resolution
    ticket_matches = matches.get("ticket_matches", [])
    if ticket_matches:
        t = ticket_matches[0]
        resolution = t.get("resolution_summary", "")
        url = f"https://test-instance.atlassian.net/browse/{t['ticket_key']}"
        lines.append(
            f"We've seen this issue before. The fix was: {resolution}"
        )
        lines.append("")

    # FAQ match — link to the article
    faq_matches = matches.get("faq_matches", [])
    if faq_matches:
        faq = faq_matches[0]
        faq_url = faq.get("confluence_url", "")
        if faq_url:
            lines.append(
                f"For step-by-step guidance, please refer to "
                f"[{faq['title']}]({faq_url})."
            )
        else:
            lines.append(
                f"We have an FAQ article on this topic: \"{faq['title']}\"."
            )
        lines.append("")

    # KB match — additional reference
    kb_matches = matches.get("kb_matches", [])
    if kb_matches and not faq_matches:
        kb = kb_matches[0]
        kb_url = kb.get("url", "")
        if kb_url:
            lines.append(
                f"You can find more details in our knowledge base: "
                f"[{kb['title']}]({kb_url})."
            )
            lines.append("")

    # Atlassian docs fallback — vendor reference
    atlassian_matches = matches.get("atlassian_matches", [])
    if atlassian_matches:
        doc = atlassian_matches[0]
        lines.append(
            f"Atlassian also has official documentation on this topic: "
            f"[{doc['title']}]({doc['url']}). Note that our internal "
            f"procedures may differ slightly from the vendor docs."
        )
        lines.append("")

    # Closing if partial coverage
    if not ticket_matches and not faq_matches:
        lines.append(
            "I'll keep looking into this and follow up with more specific guidance."
        )

    return "\n".join(lines).strip()


def run_simulation():
    """Run all fake tickets through lookup → draft response."""
    from faq.lookup import lookup

    results = []
    for ticket in FAKE_TICKETS:
        # Lookup uses the summary (same as auto_responder.py does)
        matches = lookup(ticket["summary"])

        # If no matches on summary, try with description keywords (same fallback as auto_responder)
        if not matches["found"] and ticket["description"]:
            words = " ".join(ticket["description"].split()[:20])
            matches = lookup(words)

        draft = draft_technician_response(ticket, matches)
        best_link = _pick_best_link(matches, ticket["summary"])

        results.append({
            "ticket": ticket,
            "matches": matches,
            "draft": draft,
            "best_link": best_link,
        })

    return results


def write_report(results):
    """Write the simulation report as markdown."""
    lines = [
        "# Ticket Response Simulation",
        "",
        f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Tickets processed:** {len(results)}",
        "",
        "Simulates a technician receiving tickets and the system drafting a "
        "response with inline links to relevant articles. No Gemini API calls — "
        "responses are templated from lookup results to show what the pipeline "
        "would produce.",
        "",
        "## Seed Data",
        "",
        f"| Source | Count |",
        f"|--------|-------|",
        f"| Atlassian doc URLs (sitemap-indexed) | {len(ATLASSIAN_DOCS)} |",
        f"| FAQ entries | {len(FAQ_ENTRIES)} |",
        f"| KB articles | {len(KB_ARTICLES)} |",
        f"| Resolved tickets | {len(RESOLVED_TICKETS)} |",
        "",
        "---",
        "",
    ]

    found = sum(1 for r in results if r["matches"]["found"])
    with_link = sum(1 for r in results if r["best_link"])
    lines.extend([
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Tickets with matches | {found} / {len(results)} |",
        f"| Tickets with a linkable article | {with_link} / {len(results)} |",
        f"| No match (needs manual investigation) | {len(results) - found} |",
        "",
        "---",
        "",
    ])

    for i, r in enumerate(results, 1):
        ticket = r["ticket"]
        matches = r["matches"]
        draft = r["draft"]
        best = r["best_link"]

        found_str = "MATCH" if matches["found"] else "NO MATCH"

        lines.extend([
            f"## Ticket {i}: {ticket['key']} — {found_str}",
            "",
            f"**Summary:** {ticket['summary']}",
            "",
            f"**Customer wrote:**",
            f"> {ticket['description']}",
            "",
        ])

        # What the system found
        source_parts = []
        if matches["faq_matches"]:
            titles = [f['title'] for f in matches['faq_matches']]
            source_parts.append(f"FAQ: {titles[0]}")
        if matches["kb_matches"]:
            titles = [k['title'] for k in matches['kb_matches']]
            source_parts.append(f"KB: {titles[0]}")
        if matches["ticket_matches"]:
            keys = [t['ticket_key'] for t in matches['ticket_matches']]
            source_parts.append(f"Resolved: {keys[0]}")
        if matches["atlassian_matches"]:
            titles = [d['title'] for d in matches['atlassian_matches']]
            source_parts.append(f"Atlassian Docs: {titles[0]}")

        if source_parts:
            lines.append("**Sources matched:**")
            for s in source_parts:
                lines.append(f"- {s}")
            lines.append("")

        if best:
            lines.append(f"**Best link:** [{best[0]}]({best[1]})")
            lines.append("")

        lines.extend([
            "**Draft response to customer:**",
            "",
            draft,
            "",
            "---",
            "",
        ])

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    return OUTPUT_FILE


def main():
    tmp_dir = tempfile.mkdtemp()
    tmp_db = os.path.join(tmp_dir, "sim_test.db")

    with patch("db.DB_PATH", tmp_db), \
         patch("faq.lookup.CLOUD_URL", "https://test-instance.atlassian.net"):

        from db import init_db
        init_db()

        print("Seeding database with test data...")
        seed_database()

        print(f"Running {len(FAKE_TICKETS)} fake tickets through lookup pipeline...\n")
        results = run_simulation()

        output_path = write_report(results)

        # Console summary
        for i, r in enumerate(results, 1):
            ticket = r["ticket"]
            m = r["matches"]
            best = r["best_link"]
            status = "MATCH   " if m["found"] else "NO MATCH"

            print(f"  [{status}] {ticket['key']}: {ticket['summary']}")

            if best:
                print(f"             → {best[0]}")
                print(f"               {best[1]}")
            elif not m["found"]:
                print(f"             → (no matching articles found)")
            print()

        found = sum(1 for r in results if r["matches"]["found"])
        print(f"Results: {found}/{len(results)} tickets matched to articles")
        print(f"Report:  {output_path}")

    try:
        os.remove(tmp_db)
        os.rmdir(tmp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
