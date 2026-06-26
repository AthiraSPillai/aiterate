from __future__ import annotations

import os

from aiterate.config import settings
from aiterate.tracking.base import NoopTracker


class LangSmithTracker(NoopTracker):
    """Placeholder plugin surface for LangSmith tracing.

    The class intentionally no-ops until a project supplies LangSmith credentials and trace wiring.
    Keeping it behind the common Tracker contract lets teams swap it in without changing optimizer
    code.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        project: str | None = None,
    ) -> None:
        endpoint_url = endpoint_url or settings.langsmith_endpoint
        api_key = api_key or settings.langsmith_api_key
        if endpoint_url:
            os.environ["LANGSMITH_ENDPOINT"] = endpoint_url
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
        if project:
            os.environ["LANGSMITH_PROJECT"] = project
