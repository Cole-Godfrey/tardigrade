from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        from tardigrade.dashboard import Dashboard

        Dashboard().start()
        return

    print("Usage: tardigrade dashboard")
    raise SystemExit(1)
