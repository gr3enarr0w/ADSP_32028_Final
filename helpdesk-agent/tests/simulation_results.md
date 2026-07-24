# Ticket Response Simulation

**Run:** 2026-03-15 19:23:43
**Tickets processed:** 10

Simulates a technician receiving tickets and the system drafting a response with inline links to relevant articles. No Gemini API calls — responses are templated from lookup results to show what the pipeline would produce.

## Seed Data

| Source | Count |
|--------|-------|
| Atlassian doc URLs (sitemap-indexed) | 15 |
| FAQ entries | 4 |
| KB articles | 3 |
| Resolved tickets | 6 |

---

## Summary

| Metric | Count |
|--------|-------|
| Tickets with matches | 9 / 10 |
| Tickets with a linkable article | 9 / 10 |
| No match (needs manual investigation) | 1 |

---

## Ticket 1: <PROJECT_KEY>-2001 — MATCH

**Summary:** SLA timers all say paused since we moved to cloud

**Customer wrote:**
> Hi, since we migrated to Jira Cloud last week all of our SLA timers are showing as 'Paused' even on active tickets. We have 3 SLA policies set up — Time to First Response, Time to Resolution, and Ongoing. None of them are counting down. Our team is getting dinged on SLA compliance and we need this fixed ASAP.

**Sources matched:**
- FAQ: How to reconfigure SLAs after DC-to-Cloud migration
- KB: Migration Checklist for Jira DC to Cloud
- Resolved: <PROJECT_KEY>-1001

**Best link:** [SLA not working after migration](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1001)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "SLA timers all say paused since we moved to cloud" and I have some information that should help.

We've seen this issue before. The fix was: Reconfigured SLA calendar to use Cloud business hours. The DC calendar format was not compatible. Reset SLA on affected tickets via bulk change.

For step-by-step guidance, please refer to [How to reconfigure SLAs after DC-to-Cloud migration](https://wiki.example.com/display/FAQ/SLA+Reconfiguration).

---

## Ticket 2: <PROJECT_KEY>-2002 — MATCH

**Summary:** Dashboard broken - all gadgets show error

**Customer wrote:**
> My main dashboard that I use every day is completely broken after the migration. Every gadget says 'Unable to load gadget' or just shows a grey box. I had a pie chart for issue types, a filter results gadget for my queue, and a two-dimensional filter. Can you help me get these back?

**Sources matched:**
- KB: Known Issues After Cloud Migration
- Resolved: <PROJECT_KEY>-1003

**Best link:** [Dashboard gadgets missing after migration](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1003)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Dashboard broken - all gadgets show error" and I have some information that should help.

We've seen this issue before. The fix was: Some DC gadgets have no Cloud equivalent. Replaced with Cloud-native gadgets: Pie Chart, Filter Results, and Two-Dimensional Filter.

You can find more details in our knowledge base: [Known Issues After Cloud Migration](https://wiki.example.com/display/MIGRATE/Known+Issues).

---

## Ticket 3: <PROJECT_KEY>-2003 — MATCH

**Summary:** Can't add our dev team group as assignee

**Customer wrote:**
> We're trying to assign tickets to our development team group but the group doesn't show up in the assignee field. The group name is 'platform-engineering' and it existed in DC. I can see it in admin.atlassian.com under groups but it doesn't appear in the project.

**Sources matched:**
- Resolved: <PROJECT_KEY>-1002

**Best link:** [Cannot assign tickets to group](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1002)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Can't add our dev team group as assignee" and I have some information that should help.

We've seen this issue before. The fix was: Group was migrated but not added to the project role. Added the group to the 'Service Desk Team' project role in Cloud project settings.

---

## Ticket 4: <PROJECT_KEY>-2004 — MATCH

**Summary:** automation rule to set priority isn't working anymore

**Customer wrote:**
> We had an automation rule in DC that set ticket priority to Critical when the customer type is VIP. After migration the rule shows as enabled but when a VIP submits a ticket the priority stays at Medium. I checked the rule log and it says 'action failed' but no details.

**Sources matched:**
- FAQ: Troubleshooting automation rules after Cloud migration
- Resolved: <PROJECT_KEY>-1006

**Best link:** [Automation rule runs but does not update field](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1006)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "automation rule to set priority isn't working anymore" and I have some information that should help.

We've seen this issue before. The fix was: The automation rule was using a DC-only smart value syntax. Updated to Cloud-compatible syntax: {{issue.fields.customfield_10001}} to {{issue.customfield_10001}}.

For step-by-step guidance, please refer to [Troubleshooting automation rules after Cloud migration](https://wiki.example.com/display/FAQ/Automation+Troubleshooting).

---

## Ticket 5: <PROJECT_KEY>-2005 — MATCH

**Summary:** Custom fields missing from ticket create screen

**Customer wrote:**
> When I create a new ticket in our migrated Cloud project, several custom fields are gone. Specifically 'Environment', 'Business Unit', and 'Impact Assessment' which are required fields for our process. The fields exist in the system but aren't on the create screen.

**Sources matched:**
- FAQ: Resolving custom field problems after migration
- KB: Known Issues After Cloud Migration
- Resolved: <PROJECT_KEY>-1003

**Best link:** [Dashboard gadgets missing after migration](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1003)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Custom fields missing from ticket create screen" and I have some information that should help.

We've seen this issue before. The fix was: Some DC gadgets have no Cloud equivalent. Replaced with Cloud-native gadgets: Pie Chart, Filter Results, and Two-Dimensional Filter.

For step-by-step guidance, please refer to [Resolving custom field problems after migration](https://wiki.example.com/display/FAQ/Custom+Field+Issues).

---

## Ticket 6: <PROJECT_KEY>-2006 — MATCH

**Summary:** Customers not receiving any emails from service desk

**Customer wrote:**
> Multiple customers have reported they aren't getting notification emails when we comment on their tickets or when status changes. This started right after the cloud migration. I checked their notification preferences and everything looks enabled.

**Sources matched:**
- KB: Known Issues After Cloud Migration

**Best link:** [Known Issues After Cloud Migration](https://wiki.example.com/display/MIGRATE/Known+Issues)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Customers not receiving any emails from service desk" and I have some information that should help.

You can find more details in our knowledge base: [Known Issues After Cloud Migration](https://wiki.example.com/display/MIGRATE/Known+Issues).

I'll keep looking into this and follow up with more specific guidance.

---

## Ticket 7: <PROJECT_KEY>-2007 — MATCH

**Summary:** Who do I contact to get Cloud access for new hires?

**Customer wrote:**
> We have 5 new team members starting next Monday and they need access to Jira Cloud and Confluence Cloud. In the old DC system I could just add them to the LDAP group but I'm not sure how onboarding works now. What's the process?

**Sources matched:**
- FAQ: How to reconfigure SLAs after DC-to-Cloud migration
- KB: Migration Checklist for Jira DC to Cloud

**Best link:** [How to reconfigure SLAs after DC-to-Cloud migration](https://wiki.example.com/display/FAQ/SLA+Reconfiguration)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Who do I contact to get Cloud access for new hires?" and I have some information that should help.

For step-by-step guidance, please refer to [How to reconfigure SLAs after DC-to-Cloud migration](https://wiki.example.com/display/FAQ/SLA+Reconfiguration).

---

## Ticket 8: <PROJECT_KEY>-2008 — MATCH

**Summary:** Need to set up queues for our new cloud service project

**Customer wrote:**
> We're setting up our service project in Cloud and need to create queues similar to what we had in DC. We need a queue for each team (Network, Desktop, Cloud Ops) filtered by component. Can you point me to docs on how queues work in Cloud?

**Sources matched:**
- FAQ: How to reconfigure SLAs after DC-to-Cloud migration
- KB: Migration Checklist for Jira DC to Cloud

**Best link:** [How to reconfigure SLAs after DC-to-Cloud migration](https://wiki.example.com/display/FAQ/SLA+Reconfiguration)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Need to set up queues for our new cloud service project" and I have some information that should help.

For step-by-step guidance, please refer to [How to reconfigure SLAs after DC-to-Cloud migration](https://wiki.example.com/display/FAQ/SLA+Reconfiguration).

---

## Ticket 9: <PROJECT_KEY>-2009 — MATCH

**Summary:** Workflow transition buttons disappeared

**Customer wrote:**
> After migration, the transition buttons on our tickets look different. In DC we had custom transition screens that asked for a resolution comment and a root cause field. Now when I click 'Resolve' it just resolves immediately without asking for any info. We need those screens back.

**Sources matched:**
- KB: Known Issues After Cloud Migration
- Resolved: <PROJECT_KEY>-1004

**Best link:** [Workflow transition screen not showing custom fields](https://test-instance.atlassian.net/browse/<PROJECT_KEY>-1004)

**Draft response to customer:**

Thanks for reaching out. I've taken a look at your issue regarding "Workflow transition buttons disappeared" and I have some information that should help.

We've seen this issue before. The fix was: Transition screens needed to be re-associated in Cloud. The screen scheme mapping was not preserved during migration. Reconfigured in Project Settings > Screens.

You can find more details in our knowledge base: [Known Issues After Cloud Migration](https://wiki.example.com/display/MIGRATE/Known+Issues).

---

## Ticket 10: <PROJECT_KEY>-2010 — NO MATCH

**Summary:** How to connect Slack alerts to Bitbucket deployment pipeline

**Customer wrote:**
> I want to set up alerts in our #deployments Slack channel whenever a Bitbucket pipeline deploys to production. This isn't really a migration issue, just wondering if the team can help set this up in our new Cloud environment.

**Draft response to customer:**

Thanks for reaching out. I've looked into your request regarding "How to connect Slack alerts to Bitbucket deployment pipeline" but I wasn't able to find a matching article or previous resolution in our knowledge base. I'll investigate further and get back to you shortly.

---
