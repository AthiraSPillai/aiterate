from __future__ import annotations

from aiterate.tracking.base import NoopTracker


class LangSmithTracker(NoopTracker):
    """Placeholder plugin surface for LangSmith tracing.

    The class intentionally no-ops until a project supplies LangSmith credentials and trace wiring.
    Keeping it behind the common Tracker contract lets teams swap it in without changing optimizer
    code.
    """

