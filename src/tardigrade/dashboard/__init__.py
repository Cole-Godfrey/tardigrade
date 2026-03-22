from __future__ import annotations

import importlib
import threading


def _check_deps() -> None:
    try:
        importlib.import_module("textual")
    except ImportError:
        raise ImportError(
            "Dashboard requires extra dependencies. "
            "Install with: pip install tardigrade-ai[dashboard]"
        ) from None


class Dashboard:
    def __init__(self) -> None:
        _check_deps()
        from tardigrade._logging import configure_logging
        from tardigrade.dashboard._app import TardigradeDashboard

        configure_logging(enable_dashboard=True)
        self._app = TardigradeDashboard()

    def start(self) -> None:
        self._app.run()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread
