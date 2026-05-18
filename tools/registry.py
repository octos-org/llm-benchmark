"""Padding-tool registry.

Two sources combine to give us a pool of plausible-looking tools that have
nothing to do with weather:

1. ``HAND_CURATED`` -- 99 hand-written entries covering common SaaS/infra
   verbs (search, query, send, create, etc.) with realistic ~150-char
   descriptions.
2. ``_synthetic_pads`` -- generates additional entries by composing
   (domain, action) pairs. Used only when the requested N exceeds the
   hand-curated pool, so the easy slots stay realistic.

Tool *names* in the final list are all unique. Descriptions are short
enough to keep the request body manageable at N=1000.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .relevant import RELEVANT_TOOL

# Each tuple is (name, description, [(param_name, param_type, param_desc), ...])
PadSpec = Tuple[str, str, List[Tuple[str, str, str]]]


HAND_CURATED: List[PadSpec] = [
    ("search_documents", "Search internal company documents by keyword, filters, and date range.", [("query", "string", "Search query string."), ("limit", "integer", "Max results to return.")]),
    ("query_database", "Run a parameterized SQL query against the analytics warehouse and return rows.", [("sql", "string", "SQL query to execute."), ("params", "array", "Bound parameters.")]),
    ("send_email", "Send an email message to one or more recipients with optional attachments.", [("to", "string", "Recipient email."), ("subject", "string", "Subject line."), ("body", "string", "Email body.")]),
    ("generate_image", "Generate an image from a natural language prompt using a diffusion model.", [("prompt", "string", "Image prompt."), ("size", "string", "Image dimensions.")]),
    ("transcribe_audio", "Transcribe an audio file to text using a speech-to-text model.", [("url", "string", "Audio file URL."), ("language", "string", "BCP-47 language tag.")]),
    ("summarize_pdf", "Extract text from a PDF and produce a concise summary of its contents.", [("url", "string", "PDF URL."), ("max_length", "integer", "Max summary length in words.")]),
    ("extract_entities", "Run NER over input text and return entities tagged with types and offsets.", [("text", "string", "Input text."), ("types", "array", "Entity types to extract.")]),
    ("translate_text", "Translate text from one natural language to another using a neural MT system.", [("text", "string", "Source text."), ("target_lang", "string", "Target language code.")]),
    ("geocode_address", "Convert a postal address into geographic coordinates (latitude, longitude).", [("address", "string", "Postal address."), ("country", "string", "Country hint.")]),
    ("book_meeting", "Schedule a calendar meeting with the given attendees, time, and location.", [("title", "string", "Meeting title."), ("attendees", "array", "List of attendee emails."), ("start", "string", "ISO 8601 start time.")]),
    ("create_jira_ticket", "Create a new Jira issue with summary, description, project, and priority.", [("project", "string", "Project key."), ("summary", "string", "Ticket summary."), ("priority", "string", "Priority level.")]),
    ("update_jira_ticket", "Update fields on an existing Jira issue, including status and assignee.", [("key", "string", "Issue key."), ("fields", "object", "Fields to update.")]),
    ("search_github_repos", "Search GitHub repositories by query string, language, and stars filter.", [("query", "string", "Search query."), ("language", "string", "Language filter.")]),
    ("get_stock_quote", "Fetch the latest stock price quote for a ticker symbol from the market feed.", [("symbol", "string", "Ticker symbol."), ("exchange", "string", "Exchange code.")]),
    ("compute_route", "Compute a driving or walking route between two geographic points.", [("from", "string", "Origin address or coords."), ("to", "string", "Destination address or coords."), ("mode", "string", "Travel mode.")]),
    ("search_flights", "Search for available flights between two airports on a given date.", [("origin", "string", "Origin IATA code."), ("destination", "string", "Destination IATA code."), ("date", "string", "Departure date.")]),
    ("book_hotel", "Search and reserve a hotel room for given dates, location, and guest count.", [("city", "string", "City name."), ("checkin", "string", "Check-in date."), ("nights", "integer", "Number of nights.")]),
    ("convert_currency", "Convert an amount from one currency to another using current exchange rates.", [("amount", "number", "Amount to convert."), ("from", "string", "Source currency."), ("to", "string", "Target currency.")]),
    ("send_slack_message", "Post a message to a Slack channel or direct message conversation.", [("channel", "string", "Channel ID or name."), ("text", "string", "Message text.")]),
    ("create_calendar_event", "Add an event to the user's calendar with optional reminders and attendees.", [("title", "string", "Event title."), ("start", "string", "ISO 8601 start time."), ("end", "string", "ISO 8601 end time.")]),
    ("list_files_in_drive", "List files in a Google Drive folder, optionally filtered by mime type.", [("folder_id", "string", "Drive folder ID."), ("mime_type", "string", "Mime type filter.")]),
    ("upload_to_s3", "Upload a local file to an Amazon S3 bucket with the given key.", [("path", "string", "Local file path."), ("bucket", "string", "S3 bucket."), ("key", "string", "Object key.")]),
    ("download_from_s3", "Download an object from an S3 bucket to a local destination path.", [("bucket", "string", "S3 bucket."), ("key", "string", "Object key."), ("dest", "string", "Local destination path.")]),
    ("invoke_lambda", "Invoke an AWS Lambda function synchronously with a JSON payload.", [("function", "string", "Function name."), ("payload", "object", "JSON payload.")]),
    ("publish_sns", "Publish a notification message to an SNS topic for fan-out delivery.", [("topic_arn", "string", "SNS topic ARN."), ("message", "string", "Message body.")]),
    ("send_sqs", "Enqueue a message into an SQS queue for asynchronous processing.", [("queue_url", "string", "SQS queue URL."), ("message", "string", "Message body.")]),
    ("run_shell_command", "Execute a shell command in the sandboxed environment and return its output.", [("command", "string", "Command to run."), ("cwd", "string", "Working directory.")]),
    ("read_file", "Read the contents of a file from the local filesystem and return as a string.", [("path", "string", "Absolute file path.")]),
    ("write_file", "Write a string value to a file on the local filesystem, creating it if needed.", [("path", "string", "Absolute file path."), ("content", "string", "File contents.")]),
    ("list_directory", "List the entries of a directory, optionally filtered by glob pattern.", [("path", "string", "Directory path."), ("pattern", "string", "Glob filter.")]),
    ("grep_files", "Search for a regular expression pattern across files in a directory tree.", [("pattern", "string", "Regex pattern."), ("path", "string", "Search root.")]),
    ("query_dns", "Resolve a hostname to its DNS records (A, AAAA, MX, TXT, ...).", [("hostname", "string", "Hostname."), ("type", "string", "Record type.")]),
    ("ping_host", "Send ICMP echo requests to a host and report round-trip times and packet loss.", [("host", "string", "Target host."), ("count", "integer", "Number of pings.")]),
    ("traceroute_host", "Trace the network path to a host and report each intermediate hop.", [("host", "string", "Target host.")]),
    ("send_http_request", "Send an arbitrary HTTP request with given method, URL, headers, and body.", [("method", "string", "HTTP method."), ("url", "string", "Target URL."), ("headers", "object", "Request headers.")]),
    ("parse_json", "Parse a JSON string and return the resulting object for downstream tools.", [("text", "string", "JSON text.")]),
    ("format_json", "Format a JSON value as a pretty-printed string with the given indent.", [("value", "object", "Value to format."), ("indent", "integer", "Indent width.")]),
    ("encode_base64", "Encode a binary or text payload as base64 for transport over text channels.", [("data", "string", "Input data.")]),
    ("decode_base64", "Decode a base64-encoded string back into its original binary payload.", [("data", "string", "Base64 string.")]),
    ("hash_string", "Compute a cryptographic hash digest of an input string using the given algorithm.", [("input", "string", "Input string."), ("algorithm", "string", "e.g. sha256, md5.")]),
    ("create_user", "Provision a new user account with the given email, name, and initial role.", [("email", "string", "User email."), ("name", "string", "Display name."), ("role", "string", "Initial role.")]),
    ("delete_user", "Permanently delete a user account and revoke all their active sessions.", [("user_id", "string", "User ID."), ("reason", "string", "Audit reason.")]),
    ("list_users", "List user accounts filtered by role, status, or creation date range.", [("role", "string", "Role filter."), ("limit", "integer", "Max results.")]),
    ("reset_password", "Send a password reset link to a user's registered email address.", [("user_id", "string", "User ID.")]),
    ("change_role", "Change the role assigned to a user, subject to authorization checks.", [("user_id", "string", "User ID."), ("role", "string", "New role.")]),
    ("issue_token", "Issue a short-lived API access token bound to a user and scope.", [("user_id", "string", "User ID."), ("scope", "string", "Token scope.")]),
    ("revoke_token", "Revoke a previously issued API token immediately so future calls are rejected.", [("token_id", "string", "Token identifier.")]),
    ("get_analytics_event", "Fetch a single analytics event from the event store by its ID.", [("event_id", "string", "Event ID.")]),
    ("log_analytics_event", "Record a new analytics event with type, payload, and timestamp.", [("type", "string", "Event type."), ("payload", "object", "Event payload.")]),
    ("get_user_profile", "Fetch the profile record for a user including preferences and settings.", [("user_id", "string", "User ID.")]),
    ("update_user_profile", "Apply a partial update to a user profile's preferences and settings.", [("user_id", "string", "User ID."), ("patch", "object", "Patch to apply.")]),
    ("export_data", "Trigger an export of user data into a downloadable archive (GDPR-style).", [("user_id", "string", "User ID."), ("format", "string", "Export format.")]),
    ("import_data", "Import structured data from an uploaded archive into the system.", [("archive_url", "string", "Archive URL."), ("format", "string", "Archive format.")]),
    ("trigger_workflow", "Start a workflow execution with the given workflow name and input payload.", [("workflow", "string", "Workflow name."), ("input", "object", "Input payload.")]),
    ("get_workflow_status", "Fetch the current status, progress, and last event of a workflow run.", [("run_id", "string", "Run identifier.")]),
    ("cancel_workflow", "Cancel an in-progress workflow run and clean up any orphaned resources.", [("run_id", "string", "Run identifier.")]),
    ("list_workflows", "List all known workflow definitions with their version and trigger type.", [("trigger_type", "string", "Trigger type filter.")]),
    ("publish_pubsub", "Publish a message to a Google Cloud Pub/Sub topic for downstream consumers.", [("topic", "string", "Topic name."), ("message", "string", "Message body.")]),
    ("subscribe_pubsub", "Subscribe to messages on a Pub/Sub topic and invoke a callback per message.", [("topic", "string", "Topic name."), ("callback", "string", "Callback URL.")]),
    ("query_bigquery", "Run a SQL query against a BigQuery dataset and return result rows as JSON.", [("sql", "string", "SQL query."), ("project", "string", "GCP project ID.")]),
    ("create_dataset", "Create a new dataset within a BigQuery project with the given access policy.", [("project", "string", "Project ID."), ("name", "string", "Dataset name.")]),
    ("describe_table", "Describe the schema, partitioning, and clustering of a BigQuery table.", [("project", "string", "Project ID."), ("table", "string", "Table FQN.")]),
    ("copy_object", "Copy an object from one storage bucket/key to another, preserving metadata.", [("source", "string", "Source URI."), ("destination", "string", "Destination URI.")]),
    ("delete_object", "Permanently delete an object from cloud storage and purge any replicas.", [("uri", "string", "Object URI.")]),
    ("set_object_metadata", "Update metadata (content-type, cache-control, custom keys) on a storage object.", [("uri", "string", "Object URI."), ("metadata", "object", "Metadata dict.")]),
    ("monitor_metric", "Query a time-series metric and return data points for the requested window.", [("metric", "string", "Metric name."), ("window", "string", "Time window.")]),
    ("create_alert", "Configure an alerting rule that triggers when a metric crosses a threshold.", [("metric", "string", "Metric to watch."), ("threshold", "number", "Numeric threshold.")]),
    ("list_alerts", "List currently configured alerting rules and their notification channels.", [("severity", "string", "Severity filter.")]),
    ("acknowledge_alert", "Acknowledge an active alert, suppressing further notifications for a window.", [("alert_id", "string", "Alert ID."), ("duration", "string", "Suppression window.")]),
    ("create_dashboard", "Create a new metrics dashboard with the given widget layout and queries.", [("name", "string", "Dashboard name."), ("widgets", "array", "Widget definitions.")]),
    ("share_dashboard", "Share a dashboard with another user or group, with view or edit permission.", [("dashboard_id", "string", "Dashboard ID."), ("user_id", "string", "Target user.")]),
    ("create_secret", "Store a new secret value in the secrets manager and return its identifier.", [("name", "string", "Secret name."), ("value", "string", "Secret value.")]),
    ("read_secret", "Fetch the value of a stored secret by its identifier, subject to ACL checks.", [("name", "string", "Secret name.")]),
    ("rotate_secret", "Rotate a stored secret to a new value and update any dependent integrations.", [("name", "string", "Secret name.")]),
    ("list_secrets", "List the names and metadata of secrets visible to the caller.", [("prefix", "string", "Name prefix filter.")]),
    ("encrypt_blob", "Encrypt a binary blob with a named key from the key management service.", [("key_id", "string", "KMS key ID."), ("plaintext", "string", "Plaintext bytes.")]),
    ("decrypt_blob", "Decrypt a previously encrypted blob using the matching KMS key.", [("key_id", "string", "KMS key ID."), ("ciphertext", "string", "Ciphertext.")]),
    ("sign_payload", "Sign a payload with a private key from the KMS, returning a detached signature.", [("key_id", "string", "Signing key ID."), ("payload", "string", "Payload to sign.")]),
    ("verify_signature", "Verify a signature against a payload using the matching public key.", [("key_id", "string", "Signing key ID."), ("payload", "string", "Original payload."), ("signature", "string", "Signature bytes.")]),
    ("schedule_task", "Schedule a task to run at a specific time or on a recurring cron schedule.", [("task", "string", "Task name."), ("schedule", "string", "Cron expression or ISO time.")]),
    ("cancel_task", "Cancel a previously scheduled task, removing it from the queue.", [("task_id", "string", "Task ID.")]),
    ("list_tasks", "List scheduled and recently executed tasks with their status and last run.", [("status", "string", "Status filter.")]),
    ("rate_limit_check", "Check whether a given action is currently allowed under the rate-limit policy.", [("action", "string", "Action name."), ("subject", "string", "Subject ID.")]),
    ("rate_limit_consume", "Consume one rate-limit token for the given action and subject, if available.", [("action", "string", "Action name."), ("subject", "string", "Subject ID.")]),
    ("feature_flag_get", "Look up the current value of a feature flag for a user or environment.", [("flag", "string", "Flag name."), ("user_id", "string", "User context.")]),
    ("feature_flag_set", "Set or override the value of a feature flag for an environment or user.", [("flag", "string", "Flag name."), ("value", "boolean", "New value.")]),
    ("get_billing_balance", "Fetch the current billing balance and credit status for an account.", [("account_id", "string", "Account ID.")]),
    ("charge_credit_card", "Charge an amount to a stored credit card on file for an account.", [("account_id", "string", "Account ID."), ("amount", "number", "Amount in cents.")]),
    ("issue_refund", "Issue a refund of a specified amount against an earlier charge.", [("charge_id", "string", "Charge ID."), ("amount", "number", "Refund amount.")]),
    ("subscribe_plan", "Subscribe an account to a billing plan, starting immediately or on a future date.", [("account_id", "string", "Account ID."), ("plan", "string", "Plan code.")]),
    ("cancel_subscription", "Cancel an active subscription, with optional end-of-period vs immediate semantics.", [("subscription_id", "string", "Subscription ID."), ("immediate", "boolean", "Immediate cancel.")]),
    ("send_sms", "Send an SMS text message to a phone number via the configured SMS gateway.", [("phone", "string", "E.164 phone number."), ("text", "string", "Message text.")]),
    ("send_push", "Send a mobile push notification to a registered device token.", [("device_token", "string", "Device token."), ("title", "string", "Notification title."), ("body", "string", "Notification body.")]),
    ("send_whatsapp", "Send a WhatsApp message to a phone number via the business messaging API.", [("phone", "string", "Phone number."), ("text", "string", "Message text.")]),
    ("create_invoice", "Create a new invoice for an account with line items and a due date.", [("account_id", "string", "Account ID."), ("items", "array", "Line items."), ("due", "string", "Due date.")]),
    ("get_invoice", "Fetch an invoice by ID, including line items and current payment status.", [("invoice_id", "string", "Invoice ID.")]),
    ("list_invoices", "List recent invoices for an account, filtered by status and date range.", [("account_id", "string", "Account ID."), ("status", "string", "Status filter.")]),
    ("send_invoice", "Send an invoice to the customer via email with payment instructions attached.", [("invoice_id", "string", "Invoice ID.")]),
    ("create_support_ticket", "Open a new customer support ticket with subject, description, and priority.", [("subject", "string", "Ticket subject."), ("body", "string", "Description."), ("priority", "string", "Priority.")]),
    ("close_support_ticket", "Mark a support ticket as resolved, recording the resolution summary.", [("ticket_id", "string", "Ticket ID."), ("resolution", "string", "Resolution summary.")]),
    ("get_kb_article", "Fetch a knowledge-base article by ID with current content and version metadata.", [("article_id", "string", "Article ID.")]),
    ("search_kb", "Search the knowledge base for articles matching a query, scored by relevance.", [("query", "string", "Search query.")]),
]


_SYNTH_DOMAINS = [
    ("crm", "customer relationship management"),
    ("erp", "enterprise resource planning"),
    ("hr", "human resources"),
    ("logistics", "shipping and logistics"),
    ("inventory", "warehouse inventory"),
    ("marketing", "marketing campaigns"),
    ("ads", "advertising platform"),
    ("analytics", "product analytics"),
    ("kyc", "know-your-customer compliance"),
    ("auth", "authentication and identity"),
    ("billing", "billing and invoicing"),
    ("storage", "object storage"),
    ("compute", "compute orchestration"),
    ("network", "networking infrastructure"),
    ("vpn", "VPN connectivity"),
    ("dns", "DNS records"),
    ("cdn", "content delivery network"),
    ("backup", "backup and restore"),
    ("monitor", "observability metrics"),
    ("logging", "log aggregation"),
    ("audit", "security audit trail"),
    ("policy", "access policy"),
    ("git", "git repository"),
    ("ci", "continuous integration"),
    ("registry", "container registry"),
    ("ml", "machine learning model"),
    ("dataset", "machine learning dataset"),
    ("ticket", "issue tracking"),
    ("escalation", "incident escalation"),
    ("oncall", "on-call rotation"),
    ("status", "status page"),
    ("doc", "documentation page"),
    ("translation", "translation memory"),
    ("review", "code review"),
    ("rollout", "feature rollout"),
    ("ab_test", "A/B experiment"),
    ("queue", "message queue"),
    ("topic", "pubsub topic"),
    ("schema", "data schema registry"),
    ("contract", "API contract"),
    ("survey", "user survey"),
    ("nps", "net promoter score campaign"),
    ("voice", "voice call campaign"),
    ("video", "video meeting"),
    ("webinar", "webinar registration"),
    ("course", "online course enrollment"),
    ("badge", "skill badge"),
    ("compliance", "compliance policy"),
    ("tax", "tax reporting"),
    ("payroll", "payroll processing"),
    ("benefit", "employee benefits"),
    ("expense", "expense report"),
    ("vendor", "vendor management"),
    ("contract_doc", "legal contract document"),
    ("nda", "non-disclosure agreement"),
    ("license", "software license"),
    ("brand_asset", "brand asset library"),
    ("locale", "localization locale"),
    ("region", "deployment region"),
    ("incident", "production incident"),
    ("postmortem", "incident postmortem"),
    ("forecast", "demand forecast"),
    ("budget", "budget allocation"),
    ("supplier", "supplier directory"),
    ("warehouse", "warehouse facility"),
    ("inventory_lot", "inventory lot"),
    ("shipment", "outbound shipment"),
    ("carrier", "shipping carrier"),
    ("return", "return merchandise authorization"),
    ("refund_case", "refund case"),
    ("fraud_signal", "fraud detection signal"),
    ("identity_doc", "identity document"),
    ("kyb", "know-your-business profile"),
    ("sanctions", "sanctions screening result"),
    ("aml_case", "anti-money-laundering case"),
    ("compliance_audit", "compliance audit"),
    ("legal_hold", "legal hold notice"),
    ("retention_policy", "data retention policy"),
    ("encryption_key", "encryption key"),
    ("dlp_alert", "data loss prevention alert"),
    ("siem_event", "SIEM correlation event"),
    ("vuln", "vulnerability finding"),
    ("patch", "security patch"),
    ("cve", "CVE record"),
    ("threat_intel", "threat intelligence feed"),
    ("honeypot", "honeypot sensor"),
    ("dast", "dynamic application security scan"),
    ("sast", "static application security scan"),
    ("sbom", "software bill of materials"),
    ("artifact", "build artifact"),
    ("pipeline", "CI/CD pipeline"),
    ("runner", "CI runner"),
    ("cluster", "compute cluster"),
    ("namespace", "kubernetes namespace"),
    ("pod", "kubernetes pod"),
    ("ingress", "kubernetes ingress"),
    ("service_account", "service account"),
    ("iam_role", "IAM role"),
    ("permission_set", "permission set"),
    ("federation", "identity federation"),
    ("sso_config", "single sign-on configuration"),
]

_SYNTH_ACTIONS: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    ("get", "Fetch a single", [("id", "string", "Identifier.")]),
    ("list", "List all", [("filter", "string", "Filter expression."), ("limit", "integer", "Result limit.")]),
    ("create", "Create a new", [("name", "string", "Name."), ("payload", "object", "Initial payload.")]),
    ("update", "Apply a patch to an existing", [("id", "string", "Identifier."), ("patch", "object", "Patch object.")]),
    ("delete", "Permanently delete a", [("id", "string", "Identifier.")]),
    ("archive", "Archive an existing", [("id", "string", "Identifier.")]),
    ("restore", "Restore a previously archived", [("id", "string", "Identifier.")]),
    ("describe", "Describe the configuration of a", [("id", "string", "Identifier.")]),
    ("clone", "Clone an existing", [("id", "string", "Source identifier."), ("name", "string", "New name.")]),
    ("export", "Export a copy of a", [("id", "string", "Identifier."), ("format", "string", "Export format.")]),
]


def _synthetic_pads(count: int) -> List[PadSpec]:
    """Generate up to ``count`` synthetic pad specs.

    Names use ``{domain}_{action}_record`` to avoid collisions with the
    hand-curated set. Descriptions are kept short (~120 chars) so they stay
    plausible without bloating the request body.
    """
    out: List[PadSpec] = []
    for d_name, d_label in _SYNTH_DOMAINS:
        for a_verb, a_desc, a_params in _SYNTH_ACTIONS:
            if len(out) >= count:
                return out
            tname = f"{d_name}_{a_verb}_record"
            tdesc = (
                f"{a_desc} record in the {d_label} system, with full "
                "audit logging and access control."
            )
            out.append((tname, tdesc, list(a_params)))
    return out


def _build_pad_pool(needed: int) -> List[PadSpec]:
    """Return at least ``needed`` pad specs (hand-curated first)."""
    if needed <= len(HAND_CURATED):
        return HAND_CURATED[:needed]
    extra = needed - len(HAND_CURATED)
    return HAND_CURATED + _synthetic_pads(extra)


def _spec_to_tool(spec: PadSpec) -> Dict[str, Any]:
    name, desc, params = spec
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    pname: {"type": ptype, "description": pdesc}
                    for pname, ptype, pdesc in params
                },
                "required": [params[0][0]] if params else [],
            },
        },
    }


def build_tools(n: int, position: str = "first") -> List[Dict[str, Any]]:
    """Assemble a list of ``n`` tool definitions in OpenAI function format.

    Exactly one entry is ``get_weather``; the remaining ``n - 1`` are
    plausible-but-irrelevant pads. ``position`` controls where the
    relevant tool sits in the list.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    pads = [_spec_to_tool(s) for s in _build_pad_pool(n - 1)]
    if position == "first":
        tools = [RELEVANT_TOOL, *pads]
    elif position == "last":
        tools = [*pads, RELEVANT_TOOL]
    elif position == "middle":
        mid = len(pads) // 2
        tools = pads[:mid] + [RELEVANT_TOOL] + pads[mid:]
    else:
        raise ValueError(f"unknown position: {position!r}")
    assert len(tools) == n, f"build_tools internal error: got {len(tools)} != {n}"
    # Sanity: no duplicate names (could confuse the model).
    names = [t["function"]["name"] for t in tools]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate tool names produced; check pad generator")
    return tools


def total_available_pads() -> int:
    """How many distinct pad specs we can generate before recycling."""
    return len(HAND_CURATED) + len(_SYNTH_DOMAINS) * len(_SYNTH_ACTIONS)
