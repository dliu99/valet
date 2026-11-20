"""Print an episode's record as a readable transcript.

    uv run python -m claude_policy.show runs/<run>/seed03/events.jsonl [chars]
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    n = int(argv[1]) if len(argv) > 1 else 300
    for line in open(argv[0]):
        e = json.loads(line)
        if e["type"] == "claude":
            print(f"C{e['turn']:>2} {e['seconds']:5.1f}s ${e['usd']:.3f} out={e['usage']['output_tokens']} | "
                  f"{e['thinking'][:n]} | {e['text'][:n]}".replace("\n", " "))
        elif e["type"] == "tool":
            print(f"   {e['name']} {json.dumps(e['input'])} err={e['is_error']} -> {str(e['result'])[:160]}")
        elif e["type"] == "jev":
            print(f"   jev {e['question']!r} p_yes={e['p_yes']} {e['seconds']}s")
        else:
            print({k: v for k, v in e.items() if k != "tool_calls"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
