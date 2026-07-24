"""Shared Google API service builder."""

import logging
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

from config import GOOGLE_SERVICE_ACCOUNT_JSON

log = logging.getLogger(__name__)


def get_google_service(api_name: str, api_version: str, scopes: list[str]):
    """Build an authenticated Google API service.

    Args:
        api_name: API name (e.g., "docs", "slides", "sheets")
        api_version: API version (e.g., "v1")
        scopes: OAuth scopes required
    """
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_JSON, scopes=scopes,
    )
    return build(api_name, api_version, credentials=creds, cache_discovery=False)
