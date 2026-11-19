"""Follow rate for train.py episode logs (team A, C2 'msg'): P(teammate v attacks e at t+1 | u named e at t, e legal for v),
vs chance E[1/#legal attacks]. Also share of messages naming an enemy. usage: python scripts/follow_rate_train.py runs/DIR ..."""
import json, re, sys
for d in sys.argv[1:]:
    hit = tot = 0; ch = 0.0; nm = nt = 0
    for l in open(f"{d}/episodes.jsonl"):
        e = json.loads(l); by = {}
        for x in e["decisions"]:
            if x.get("team", "A") != "A": continue
            by[(x["t"], x["agent"])] = x
        for (t, u), x in by.items():
            m = re.search(r'"msg"\s*:\s*"([^"]*)"', x["reply"] or "")
            if not m or not m.group(1).strip(): continue
            nm += 1; tg = re.findall(r"\be(\d)\b", m.group(1).lower())
            if not tg: continue
            nt += 1; tgt = tg[0]
            for (t2, v), y in by.items():
                if t2 != t + 1 or v == u: continue
                la = re.search(r"Legal actions:(.*)", y["prompt"]); legal = re.findall(r"attack_e(\d)", la.group(1) if la else "")
                if tgt not in legal: continue
                a = re.search(r'"action"\s*:\s*"([^"]*)"', y["reply"] or "")
                tot += 1; hit += bool(a and a.group(1) == f"attack_e{tgt}"); ch += 1 / len(legal)
    print(f"{d}: msgs {nm}, naming enemy {nt/max(nm,1):.2f}; follow {hit/max(tot,1):.2f} vs chance {ch/max(tot,1):.2f} (n={tot})")
