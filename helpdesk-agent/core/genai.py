"""Shared Gemini (Vertex AI) client factory — single instance, used by all plugins."""

import os
import logging

from config import GOOGLE_SERVICE_ACCOUNT_JSON, GEMINI_PROJECT, GEMINI_LOCATION

log = logging.getLogger(__name__)

_client = None


def get_genai_client():
    global _client
    if _client is not None:
        return _client

    from google import genai

    sa_path = GOOGLE_SERVICE_ACCOUNT_JSON
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", sa_path)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(sa_path)

    _client = genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location=GEMINI_LOCATION,
    )
    log.info("Initialized Gemini client (project=%s, location=%s)", GEMINI_PROJECT, GEMINI_LOCATION)
    return _client
