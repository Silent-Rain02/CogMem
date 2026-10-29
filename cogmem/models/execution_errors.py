"""Execution failures must not be interpreted as missing memory evidence."""
import httpx
import requests
from openai import APIError, APIConnectionError, APIStatusError
from tenacity import RetryError


class CampaignStopped(RuntimeError):
    """No further provider calls are authorized in this campaign execution."""


def retryable_provider_error(error):
    """Retry transient SDK failures only; configuration/auth errors need repair."""
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def raise_if_execution_failure(error):
    """Preserve operational failures through legacy optional-fallback handlers."""
    if isinstance(error, (CampaignStopped, APIError, RetryError,
                          httpx.HTTPError, requests.RequestException)):
        raise error
