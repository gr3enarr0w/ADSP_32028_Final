"""Labeled article pairs for FAQ dedup threshold calibration.

40 hand-labeled pairs from Atlassian DC-to-Cloud migration content:
  - 15 duplicate pairs  (paraphrased — same topic, different wording)
  - 25 distinct pairs (including 10 borderline/related topics labeled as distinct)

Source of truth for: faq/threshold_calibration CLI and tests/test_threshold_calibration.py
To add pairs: maintain label balance and re-run calibration.
"""

from __future__ import annotations

LABELED_PAIRS: list[tuple[str, str, str]] = [
    # -- DUPLICATE pairs (15) --------------------------------------------------
    # 1 -- Jira project migration steps
    (
        "To migrate a Jira project from Data Center to Cloud, install the "
        "Jira Cloud Migration Assistant from the Atlassian Marketplace. Run "
        "the pre-migration assessment to identify blocking issues. Resolve "
        "all errors, then start the migration. Monitor progress in the "
        "migration dashboard and verify data after completion.",

        "Migrating a Jira project to Cloud requires the Jira Cloud Migration "
        "Assistant, available in the Atlassian Marketplace. First, perform a "
        "pre-migration assessment that identifies any blockers. Fix the "
        "reported issues, kick off the migration, and track it via the "
        "migration dashboard. Verify the data once it finishes.",

        "duplicate",
    ),
    # 2 -- Permission scheme differences
    (
        "Permission schemes in Jira Data Center use project roles and group "
        "memberships to control access. In Cloud, permissions are managed "
        "through the admin console with organization-level controls. Custom "
        "permission schemes are migrated automatically but may require manual "
        "review to align with Cloud role structures.",

        "Jira DC controls access via permission schemes tied to project roles "
        "and groups. After migrating to Cloud, you manage permissions in the "
        "admin console at the organization level. Although custom permission "
        "schemes carry over during migration, you should review them manually "
        "because Cloud role structures differ.",

        "duplicate",
    ),
    # 3 -- Custom field migration
    (
        "Custom fields defined in Jira Data Center are migrated to Cloud "
        "with their configurations preserved. However, some custom field "
        "types from third-party apps may not have Cloud equivalents. Review "
        "the migration assessment report for any unsupported field types and "
        "plan replacements before migrating.",

        "When moving to Cloud, Jira custom fields and their configurations "
        "are transferred automatically. Be aware that certain custom field "
        "types provided by third-party plugins might lack Cloud versions. "
        "Check the assessment report for unsupported types and arrange "
        "alternative fields before you start the migration.",

        "duplicate",
    ),
    # 4 -- SSO / SAML setup
    (
        "To configure SAML single sign-on for Jira Cloud, navigate to the "
        "organization admin console. Select Security, then Identity providers. "
        "Add your SAML identity provider by entering the SSO URL, entity ID, "
        "and uploading the signing certificate. Test the connection and enable "
        "enforcement once verified.",

        "Setting up SAML SSO in Jira Cloud is done from the organization "
        "admin console under Security and Identity providers. You provide "
        "the SSO URL, entity ID, and signing certificate for your SAML "
        "identity provider. After testing the connection successfully, "
        "enforce SSO for all organization users.",

        "duplicate",
    ),
    # 5 -- Workflow migration
    (
        "Jira workflows are migrated from Data Center to Cloud using the "
        "Cloud Migration Assistant. Simple workflows transfer without issues. "
        "Complex workflows with post-functions, validators, and conditions "
        "that rely on server-side scripting may need manual adjustment after "
        "migration because ScriptRunner and other scripting apps work "
        "differently on Cloud.",

        "The Cloud Migration Assistant handles Jira workflow migration. "
        "Basic workflows migrate cleanly, but complex workflows using "
        "post-functions, validators, or conditions that depend on server-side "
        "scripting (such as ScriptRunner) often require manual changes on "
        "Cloud due to differences in how scripting apps operate.",

        "duplicate",
    ),
    # 6 -- App / plugin compatibility
    (
        "Before migrating to Atlassian Cloud, audit all installed Data Center "
        "apps and plugins for Cloud compatibility. Use the Atlassian "
        "Marketplace to check whether a Cloud version exists. For apps "
        "without Cloud equivalents, identify alternative apps or plan to "
        "replicate the functionality with built-in Cloud features.",

        "Audit your Data Center apps and plugins for Cloud compatibility "
        "before starting migration. The Atlassian Marketplace lists whether "
        "each app has a Cloud version. Where no Cloud equivalent exists, "
        "find an alternative Marketplace app or use native Cloud capabilities "
        "to replace the functionality.",

        "duplicate",
    ),
    # 7 -- Notification scheme changes
    (
        "Notification schemes in Jira Data Center let administrators control "
        "who receives email notifications for issue events. In Jira Cloud, "
        "notification schemes still exist but users also have personal "
        "notification preferences that can override scheme settings. Admins "
        "should communicate this change to users after migration.",

        "In Jira DC, notification schemes determine email notifications for "
        "issue events. Cloud keeps notification schemes, but adds personal "
        "notification preferences that users can adjust to override the "
        "scheme defaults. After migrating, inform users about this new "
        "control over their notifications.",

        "duplicate",
    ),
    # 8 -- Data backup procedures
    (
        "Before migrating from Jira Data Center to Cloud, create a full "
        "backup of your instance. Export the database, attachments directory, "
        "and application home directory. Store the backup in a secure "
        "location separate from the production server. Verify the backup "
        "by restoring it to a staging environment.",

        "Always back up your Jira Data Center instance before Cloud "
        "migration. This includes a database export, the attachments folder, "
        "and the entire application home directory. Keep the backup on a "
        "separate, secure server. Confirm the backup works by restoring "
        "it in a staging or test environment.",

        "duplicate",
    ),
    # 9 -- User provisioning
    (
        "User provisioning for Jira Cloud can be automated through SCIM "
        "with supported identity providers such as Okta, Azure AD, or "
        "OneLogin. Configure SCIM in the Atlassian organization admin "
        "console under Directory and User provisioning. Users and groups "
        "are synchronized automatically once SCIM is enabled.",

        "Jira Cloud supports automated user provisioning via SCIM. "
        "Compatible identity providers include Okta, Azure AD, and "
        "OneLogin. Set up SCIM from the organization admin console in "
        "Directory then User provisioning. When enabled, user and group "
        "data syncs automatically from your identity provider.",

        "duplicate",
    ),
    # 10 -- API differences
    (
        "Jira Data Center uses REST API v2 with basic or OAuth 1.0a "
        "authentication. Jira Cloud uses REST API v2 and v3 with OAuth 2.0 "
        "(3LO or 2LO) or API tokens. Cloud API endpoints use a different "
        "base URL format and some endpoints have different behavior or "
        "are unavailable. Review the Cloud API docs for breaking changes.",

        "The Jira DC REST API v2 authenticates with basic auth or OAuth 1.0a. "
        "On Cloud, you use REST API v2 or v3 with OAuth 2.0 (three-legged or "
        "two-legged) or API tokens. Cloud endpoints have a different base URL "
        "and some endpoints behave differently or do not exist. Always check "
        "the Cloud API documentation for breaking changes.",

        "duplicate",
    ),
    # 11 -- Confluence space migration
    (
        "Migrating Confluence spaces from Data Center to Cloud uses the "
        "Confluence Cloud Migration Assistant. Select the spaces to migrate, "
        "run the pre-migration checks, and resolve any flagged issues. Large "
        "spaces with many attachments may take several hours. Monitor the "
        "migration dashboard for status updates.",

        "Use the Confluence Cloud Migration Assistant to move spaces from DC "
        "to Cloud. Pick the spaces you want to migrate, run pre-migration "
        "checks, and address flagged problems. Expect large spaces with "
        "heavy attachments to take hours. Watch the migration dashboard for "
        "progress.",

        "duplicate",
    ),
    # 12 -- Jira board configuration
    (
        "Jira boards in Data Center are migrated to Cloud with their filter "
        "and column configurations intact. However, board-level swimlane "
        "settings and quick filters may need reconfiguration. Check each "
        "board after migration to confirm the layout matches expectations.",

        "When migrating to Cloud, Jira boards carry over their filter and "
        "column configuration. Board swimlane settings and quick filters "
        "sometimes require reconfiguration. After migration, review each "
        "board to make sure the layout is correct.",

        "duplicate",
    ),
    # 13 -- Jira automation migration
    (
        "Automation rules created with Jira Automation for Server or Data "
        "Center need to be recreated in Jira Cloud. The Cloud automation "
        "engine uses a different rule syntax. Export your existing rules "
        "for reference and rebuild them using the Cloud automation UI.",

        "Jira automation rules from Server or Data Center do not migrate "
        "directly. The Cloud automation engine has a different rule syntax, "
        "so you must recreate rules manually. Export your DC rules as "
        "reference and use the Cloud automation interface to rebuild them.",

        "duplicate",
    ),
    # 14 -- Confluence macros
    (
        "Confluence macros from Data Center plugins may not work in Cloud. "
        "Review each macro used in your spaces against the Cloud macro "
        "catalog. Built-in macros generally migrate without issues, but "
        "macros from third-party apps often need a Cloud-compatible "
        "replacement or manual content adjustment.",

        "Some Confluence DC plugin macros are unsupported in Cloud. Compare "
        "the macros in your spaces with the Cloud macro catalog. Standard "
        "built-in macros migrate smoothly, but third-party app macros "
        "usually require either a Cloud-compatible substitute or manual "
        "content edits.",

        "duplicate",
    ),
    # 15 -- Migration timeline planning
    (
        "Plan your migration timeline by assessing the size of your instance, "
        "the number of projects and spaces, and app compatibility. Schedule "
        "a maintenance window for the cutover. Run a test migration first to "
        "estimate duration and identify issues before the production run.",

        "Estimate your migration timeline based on instance size, project "
        "and space count, and app compatibility status. Allocate a "
        "maintenance window for the actual cutover. Perform a test migration "
        "beforehand to gauge timing and catch problems early.",

        "duplicate",
    ),

    # -- DISTINCT pairs (25 total: 15 clear-cut below + 10 borderline at end) ----------
    # 16 -- Billing vs SSO
    (
        "To update your Atlassian Cloud billing information, go to "
        "admin.atlassian.com and select Billing. Update your credit card "
        "details, billing address, and review your current subscription "
        "tier. Changes take effect on the next billing cycle.",

        "Configure SAML single sign-on by navigating to the organization "
        "admin console under Security and Identity providers. Provide the "
        "SSO URL and entity ID from your identity provider.",

        "distinct",
    ),
    # 17 -- User provisioning vs data backup
    (
        "Automated user provisioning through SCIM synchronizes users and "
        "groups from your identity provider to Atlassian Cloud. Supported "
        "providers include Okta, Azure AD, and OneLogin.",

        "Create a full backup of your Jira Data Center instance including "
        "the database, attachments, and home directory before starting any "
        "migration. Store backups securely off the production server.",

        "distinct",
    ),
    # 18 -- Workflow migration vs billing setup
    (
        "Complex Jira workflows with post-functions and validators that rely "
        "on scripting plugins like ScriptRunner need manual adjustment after "
        "migrating to Cloud because these plugins work differently.",

        "Review your Atlassian Cloud subscription tier and pricing on "
        "admin.atlassian.com. Enterprise plans include additional security "
        "features like audit logging and IP allowlisting.",

        "distinct",
    ),
    # 19 -- API differences vs notification schemes
    (
        "Jira Cloud REST API uses OAuth 2.0 authentication while Data Center "
        "uses basic auth or OAuth 1.0a. Cloud endpoints have a different base "
        "URL format and some DC-only endpoints do not exist.",

        "Notification schemes control which users receive email notifications "
        "for issue events. In Cloud, users can override scheme settings with "
        "personal notification preferences.",

        "distinct",
    ),
    # 20 -- Confluence macro vs Jira board
    (
        "Confluence macros from third-party Data Center plugins may not have "
        "Cloud equivalents. Review each macro against the Cloud catalog and "
        "find replacements for unsupported ones.",

        "Jira Scrum and Kanban boards migrate with their filter and column "
        "configurations intact but swimlane and quick filter settings may "
        "need reconfiguration on Cloud.",

        "distinct",
    ),
    # 21 -- Custom fields vs app compatibility
    (
        "Custom fields in Jira Data Center are migrated to Cloud with "
        "configurations preserved. Third-party custom field types may lack "
        "Cloud equivalents and need replacement.",

        "Audit all installed Atlassian Marketplace apps before migration. "
        "Check each app for a Cloud version. For apps without Cloud support, "
        "plan alternative solutions using built-in Cloud features.",

        "distinct",
    ),
    # 22 -- Permission schemes vs timeline planning
    (
        "Jira Cloud manages permissions through the admin console at the "
        "organization level. Custom permission schemes from DC are migrated "
        "but may need manual review to match Cloud role structures.",

        "Plan your migration timeline by assessing instance size and app "
        "compatibility. Schedule a maintenance window and run a test "
        "migration to estimate duration before the production cutover.",

        "distinct",
    ),
    # 23 -- Confluence space migration vs Jira automation
    (
        "Use the Confluence Cloud Migration Assistant to move spaces from "
        "Data Center. Select spaces, run pre-migration checks, and resolve "
        "any flagged issues before starting the migration.",

        "Jira automation rules from Data Center need to be recreated in "
        "Cloud because the automation engine uses a different rule syntax. "
        "Export DC rules as reference and rebuild in the Cloud UI.",

        "distinct",
    ),
    # 24 -- Data backup vs project migration
    (
        "Back up your Jira DC database, attachments, and home directory "
        "before migration. Store backups securely and verify them by "
        "restoring to a staging environment.",

        "Install the Jira Cloud Migration Assistant to migrate projects. "
        "Run the pre-migration assessment, resolve blockers, start the "
        "migration, and monitor it in the dashboard.",

        "distinct",
    ),
    # 25 -- SSO setup vs Confluence macros
    (
        "Set up SAML SSO for Jira Cloud from the admin console under "
        "Security and Identity providers. Enter the SSO URL, entity ID, "
        "and signing certificate from your identity provider.",

        "Review Confluence macros used in your spaces for Cloud "
        "compatibility. Built-in macros generally work, but third-party "
        "app macros often need Cloud replacements or content adjustments.",

        "distinct",
    ),
    # 26 -- Billing vs user provisioning
    (
        "Update your payment method and billing address in the Atlassian "
        "admin portal. Review subscription tier pricing and feature "
        "differences between Standard, Premium, and Enterprise plans.",

        "Configure SCIM user provisioning from the organization admin "
        "console to automatically sync users and groups from Okta, "
        "Azure AD, or OneLogin.",

        "distinct",
    ),
    # 27 -- Workflow vs notification
    (
        "Jira workflows with scripting post-functions need manual "
        "reconfiguration after migration because ScriptRunner operates "
        "differently on Cloud than on Data Center.",

        "Users in Jira Cloud have personal notification preferences that "
        "can override the notification scheme settings configured by "
        "administrators. Communicate this change after migration.",

        "distinct",
    ),
    # 28 -- API vs Confluence space migration
    (
        "Cloud REST API v3 introduces Atlassian Document Format for rich "
        "text fields. OAuth 2.0 replaces basic auth. Some Data Center "
        "endpoints are deprecated or removed in Cloud.",

        "The Confluence Cloud Migration Assistant moves spaces including "
        "pages, attachments, and permissions. Large spaces may take "
        "several hours to complete.",

        "distinct",
    ),
    # 29 -- Board config vs custom fields
    (
        "Jira board column configurations and filters are preserved during "
        "migration but quick filters and swimlane settings may need manual "
        "reconfiguration on Cloud.",

        "Custom fields from third-party Jira apps may not have Cloud "
        "equivalents. Check the migration assessment report for unsupported "
        "field types and plan replacements.",

        "distinct",
    ),
    # 30 -- Automation vs permission schemes
    (
        "Recreate Data Center automation rules in the Cloud automation "
        "engine. The rule syntax differs between DC and Cloud so direct "
        "import is not supported.",

        "Permission schemes in Cloud use organization-level controls in "
        "the admin console. Custom DC permission schemes migrate but may "
        "require manual alignment with Cloud role structures.",

        "distinct",
    ),

    # -- BORDERLINE pairs (10) -- related topics, genuinely different ----------
    # 31 -- Jira project migration vs Jira user migration
    (
        "Migrate Jira projects using the Cloud Migration Assistant. Select "
        "projects, run pre-migration assessment, fix blockers, and execute "
        "the migration. Monitor progress in the dashboard.",

        "Migrate Jira users by syncing your identity provider via SCIM. "
        "Map existing DC users to Cloud accounts. Deactivate unused "
        "accounts and review group memberships after the sync completes.",

        "distinct",
    ),
    # 32 -- Jira permissions vs Confluence permissions
    (
        "Jira Cloud permissions are managed through permission schemes "
        "tied to project roles at the organization level in the admin "
        "console. Custom schemes from DC are migrated automatically.",

        "Confluence Cloud permissions are set at the space level. Space "
        "admins control who can view, edit, or administer each space. "
        "Global permissions are managed separately in the admin console.",

        "distinct",
    ),
    # 33 -- Jira workflow migration vs Confluence template migration
    (
        "Jira workflows migrate via the Cloud Migration Assistant. Complex "
        "workflows with scripting post-functions may need manual adjustment "
        "due to differences in how scripting apps run on Cloud.",

        "Confluence page templates from Data Center can be recreated in "
        "Cloud using the template editor. Some template variables and "
        "user macros may not have direct Cloud equivalents.",

        "distinct",
    ),
    # 34 -- App audit for Jira vs app audit for Confluence
    (
        "Audit Jira Data Center apps for Cloud compatibility. Check the "
        "Marketplace for Cloud versions. Plan replacements for apps that "
        "lack Cloud support using built-in Jira Cloud features.",

        "Audit Confluence Data Center apps for Cloud compatibility. Some "
        "Confluence apps like Gliffy or draw.io have Cloud versions while "
        "others may require content export and manual recreation.",

        "distinct",
    ),
    # 35 -- Pre-migration assessment vs post-migration validation
    (
        "Run the pre-migration assessment using the Cloud Migration "
        "Assistant. It identifies unsupported apps, incompatible "
        "configurations, and potential data issues before you migrate.",

        "After migration, validate your data by comparing issue counts, "
        "attachment integrity, and workflow states between DC and Cloud. "
        "Run automated checks and spot-check key projects manually.",

        "distinct",
    ),
    # 36 -- Jira issue migration vs Jira attachment migration
    (
        "Jira issues including all fields, comments, and history are "
        "migrated to Cloud by the Migration Assistant. Issue keys are "
        "preserved during migration to maintain existing references.",

        "Jira attachments are migrated alongside their parent issues. "
        "Large attachments over the Cloud file size limit may fail to "
        "transfer. Check attachment size limits before migration and "
        "archive oversized files.",

        "distinct",
    ),
    # 37 -- Cloud pricing tiers vs Cloud feature comparison
    (
        "Atlassian Cloud offers Free, Standard, Premium, and Enterprise "
        "pricing tiers. Each tier has different user limits, storage "
        "quotas, and support levels. Review pricing at atlassian.com.",

        "Atlassian Cloud Premium and Enterprise include advanced features "
        "like IP allowlisting, audit logging, sandbox environments, and "
        "priority support not available on Standard or Free plans.",

        "distinct",
    ),
    # 38 -- Jira Service Management migration vs Jira Software migration
    (
        "Migrating Jira Service Management from DC to Cloud transfers "
        "service desks, queues, SLAs, and customer portal configurations. "
        "Some automation rules and custom SLA metrics may need adjustment.",

        "Jira Software project migration moves boards, sprints, backlogs, "
        "and agile configurations to Cloud. Board column mappings and "
        "estimation settings are preserved during migration.",

        "distinct",
    ),
    # 39 -- Confluence page migration vs Confluence blog migration
    (
        "Confluence pages including content, attachments, and page trees "
        "are migrated to Cloud with the Migration Assistant. Page "
        "restrictions and labels are preserved.",

        "Confluence blog posts migrate alongside space content. Blog "
        "post dates, authors, and comments are preserved. Review blog "
        "post macros for Cloud compatibility after migration.",

        "distinct",
    ),
    # 40 -- DC server sizing vs Cloud instance scaling
    (
        "Jira Data Center server sizing depends on user count, issue "
        "volume, and concurrent request load. Nodes can be added to the "
        "cluster for horizontal scaling.",

        "Atlassian Cloud scales automatically based on usage. There is "
        "no infrastructure to manage. Performance is governed by your "
        "subscription tier and Atlassian's multi-tenant architecture.",

        "distinct",
    ),
]

