from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        from agentarmor.dashboard import Dashboard

        Dashboard().start()
        return

    print("Usage: agentarmor dashboard")
    raise SystemExit(1)
