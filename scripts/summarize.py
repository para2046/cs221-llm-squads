"""Summarize episode JSONL files: win rate, return, kills, invalid-action rate (with 95% CI)."""
import glob
import json
import math
import sys

rows = []
for path in sys.argv[1:]:
    files = glob.glob(f"{path}/*.jsonl") if not path.endswith(".jsonl") else [path]
    for f in files:
        for line in open(f):
            d = json.loads(line)
            if d.get("summary"):
                rows.append(d)
n = len(rows)
if not n:
    print(json.dumps({"episodes": 0}))
    sys.exit()


def mean_ci(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return round(m, 3), round(1.96 * sd / math.sqrt(len(xs)), 3)


print(json.dumps({
    "source": sys.argv[1:], "episodes": n,
    "win_rate": mean_ci([float(r["won"]) for r in rows]),
    "return": mean_ci([r["return"] for r in rows]),
    "dead_enemies": mean_ci([r["dead_enemies"] or 0 for r in rows]),
    "dead_allies": mean_ci([r["dead_allies"] or 0 for r in rows]),
    "invalid_rate": mean_ci([r["invalid_rate"] for r in rows]),
}))
