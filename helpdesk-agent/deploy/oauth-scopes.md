# OAuth 2LO Scopes — AI Helpdesk Agent

**App name:** jsm-helpdesk-agent  
**Client ID:** <YOUR_OAUTH_CLIENT_ID>  
**Created:** 2026-04-22  
**Type:** OAuth 2.0 (2LO) — client credentials, single app, 50 granular scopes  
**Config var:** `ATLASSIAN_OAUTH_CLIENT_ID` / `ATLASSIAN_OAUTH_CLIENT_SECRET`

All products (Jira, JSM, Confluence reads and writes) are covered by this single app.
The `JSM_OAUTH_*`, `JIRA_WRITE_*`, and `CONFLUENCE_WRITE_*` env vars are not set —
the code falls back to the primary credentials for all operations.

No delete scopes are granted — `_delete_comment()` will silently no-op.

---

## Read Scopes (40)

### Jira Platform
| Scope | Purpose |
|-------|---------|
| `read:issue:jira` | Read issues via JQL and direct fetch |
| `read:issue:jira-software` | View issue estimations and estimation fields |
| `read:issue-details:jira` | View full issue detail records |
| `read:issue-status:jira` | View issue status field |
| `read:issue-link:jira` | View issue links (related, blocks, etc.) |
| `read:issue-type:jira` | View issue types |
| `read:issue.changelog:jira` | View issue change history |
| `read:issue.transition:jira` | View available status transitions |
| `read:issue.watcher:jira` | View issue watchers |
| `read:comment:jira` | Read issue comments |
| `read:comment.property:jira` | Read comment properties (sd.public.comment internal flag) |
| `read:attachment:jira` | View issue attachments |
| `read:user:jira` | View user profiles (reporter, assignee) |
| `read:email-address:jira` | View user email addresses |
| `read:project:jira` | View projects |
| `read:project-version:jira` | View project versions (affectedVersion) |
| `read:project.component:jira` | View project components |
| `read:field:jira` | View fields including customfield_10010 (request type) |
| `read:resolution:jira` | View resolution values |
| `read:status:jira` | View status names |
| `read:priority:jira` | View issue priorities |
| `read:jql:jira` | Execute JQL queries |

### Jira Service Management
| Scope | Purpose |
|-------|---------|
| `read:servicedesk:jira-service-management` | View service desks |
| `read:requesttype:jira-service-management` | View request types |
| `read:request:jira-service-management` | View service desk requests |
| `read:request.comment:jira-service-management` | View request comments (JSM visibility layer) |
| `read:request.status:jira-service-management` | View request status and transition info |
| `read:request.participant:jira-service-management` | View request participants |
| `read:request.sla:jira-service-management` | View SLA status for prioritization |
| `read:customer:jira-service-management` | View customer account info |
| `read:servicedesk.customer:jira-service-management` | View customers per service desk |
| `read:organization:jira-service-management` | View customer organizations |
| `read:queue:jira-service-management` | View JSM queues |
| `read:knowledgebase:jira-service-management` | View JSM knowledge base articles |

### Confluence
| Scope | Purpose |
|-------|---------|
| `read:content:confluence` | View content via CQL search |
| `read:space:confluence` | View spaces |
| `read:space-details:confluence` | View space metadata |
| `read:page:confluence` | View pages |
| `read:attachment:confluence` | View attachments on KB pages |
| `read:label:confluence` | View content labels |

---

## Write Scopes (10)

### Jira Platform
| Scope | Purpose |
|-------|---------|
| `write:comment:jira` | Post internal and public comments on tickets |
| `write:comment.property:jira` | Set sd.public.comment.internal flag on comments |
| `write:issue.watcher:jira` | Add watchers to issues |

### Jira Service Management
| Scope | Purpose |
|-------|---------|
| `write:request:jira-service-management` | Create and update service desk requests |
| `write:request.comment:jira-service-management` | Post comments via JSM API |
| `write:request.status:jira-service-management` | Transition request status (resolve/close) |
| `write:request.attachment:jira-service-management` | Add attachments to requests |
| `write:request.notification:jira-service-management` | Subscribe users to request notifications |
| `write:request.participant:jira-service-management` | Add participants to requests |

### Confluence
| Scope | Purpose |
|-------|---------|
| `write:page:confluence` | Publish FAQ articles to Confluence (v2 API) |
