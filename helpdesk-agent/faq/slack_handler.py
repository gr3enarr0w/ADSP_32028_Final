"""Slack slash command handler for /jsm-assist.

Restricted to configured channels. Returns ephemeral responses
(only visible to the agent who called it).
"""

import hashlib
import hmac
import logging
import os
import time

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from faq.lookup import lookup, format_not_found

log = logging.getLogger(__name__)

slack_router = APIRouter()

# Channels where /jsm-assist is allowed — must be configured via env.
# Set to the channel ID(s) of the dedicated support lookup channel.
# Example: SLACK_LOOKUP_CHANNELS=C0XXXXXXXXX
ALLOWED_CHANNELS = set(
    c.strip() for c in os.getenv("SLACK_LOOKUP_CHANNELS", "").split(",") if c.strip()
)

# Slack signing secret for request verification
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")


def _verify_slack_signature(timestamp: str, body: bytes, signature: str) -> bool:
    """Verify Slack request signature using the signing secret."""
    if not SLACK_SIGNING_SECRET:
        log.warning("SLACK_SIGNING_SECRET not set — skipping signature verification")
        return True
    # Reject requests older than 5 minutes to prevent replay attacks
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except (ValueError, TypeError):
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@slack_router.post("/jsm-assist")
async def slash_lookup(
    request: Request,
    text: str = Form(""),
    channel_id: str = Form(""),
    user_name: str = Form(""),
    command: str = Form(""),
):
    """Handle /jsm-assist slash command from Slack.

    Slack sends form-encoded POST with: text, channel_id, user_name, etc.
    Response is ephemeral (only visible to the calling agent).

    Setup in Slack App:
      Slash Command: /jsm-assist
      Request URL: https://<service>/api/slack/jsm-assist
      Description: Look up existing FAQ/KB answers for a ticket or topic
      Usage Hint: [ticket-key or topic keywords]
    """
    # Verify Slack signing secret
    slack_signature = request.headers.get("X-Slack-Signature", "")
    slack_timestamp = request.headers.get("X-Slack-Request-Timestamp", "0")
    raw_body = await request.body()
    if not _verify_slack_signature(slack_timestamp, raw_body, slack_signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    if not text.strip():
        return JSONResponse(content={
            "response_type": "ephemeral",
            "text": "Usage: `/jsm-assist <PROJECT_KEY>-1234` or `/jsm-assist service account login`",
        })

    if ALLOWED_CHANNELS and channel_id not in ALLOWED_CHANNELS:
        return JSONResponse(content={
            "response_type": "ephemeral",
            "text": "This command is only available in designated support channels.",
        })

    log.info("Slack /jsm-assist from %s in %s: %s", user_name, channel_id, text.strip())

    import asyncio
    result = await asyncio.to_thread(lookup, text.strip())

    if result["found"]:
        response_text = result["response_draft"]
    else:
        response_text = format_not_found(text.strip())

    return JSONResponse(content={
        "response_type": "ephemeral",
        "text": response_text,
    })
