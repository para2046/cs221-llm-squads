"""Message taxonomy for C2 games (unified 'msg' field): one category per message, first match wins.
usage: python scripts/msg_taxonomy.py runs/DIR [runs/DIR ...]"""
import glob, json, re, sys, collections
CATS = [  # (name, regex) checked in order
    ("backup request / offer", r"\b(help|need|assist|support|backup|cover me|join(ing)? (fight|team|you)|coming to)\b"),
    ("regroup / wait", r"\b(regroup|group(ing)? up|gather|wait|together|hold)\b"),
    ("retreat / low hp", r"\b(retreat\w*|fall(ing)? back|back(ing)? off|low (hp|health|shield)|recover\w*|dying)\b"),
    ("kiting", r"\bkit(e|ing)\b"),
    ("kill confirm / finishing", r"\b(finish\w*|kill(ed)?|down|dead|last hit)\b"),
    ("focus call (target named)", r"\be\d\b"),
    ("focus call (target by type)", r"\b(focus\w*|target\w*|attack\w*|burn\w*|fire)\b"),
    ("movement / scouting", r"\b(mov\w*|head\w*|go\w*|advanc\w*|approach\w*|search\w*|scout\w*|north|south|east|west|closing)\b"),
]
TEAM = {"backup request / offer", "regroup / wait", "kill confirm / finishing", "focus call (target named)", "focus call (target by type)"}
def cat(m):
    t = m.lower()
    for n, r in CATS:
        if re.search(r, t):
            return n
    return "other"
for d in sys.argv[1:]:
    C, W, L = collections.Counter(), collections.Counter(), collections.Counter(); n = nm = 0; ex = collections.defaultdict(collections.Counter)
    won = {}
    for f in glob.glob(f"{d}/w*.jsonl") + glob.glob(f"{d}/seed*.jsonl"):
        rows = [json.loads(l) for l in open(f)]
        won.update({(f, r["episode"]): r["won"] for r in rows if r.get("summary")})
        for r in rows:
            if r.get("summary"): continue
            for a in r["team_a"]:
                n += 1; m = (a.get("msg") or "").strip()
                if not m: continue
                nm += 1; c = cat(m); C[c] += 1; ex[c][m.lower()] += 1
                (W if won.get((f, r["episode"])) else L)[c] += 1
    if n == 0 and glob.glob(f"{d}/episodes.jsonl"):   # train.py format (4B evals)
        for i, l in enumerate(open(f"{d}/episodes.jsonl")):
            e = json.loads(l); w = bool(e["info"]["battle_won"])
            for x in e["decisions"]:
                if x.get("team", "A") != "A": continue
                n += 1; mm = re.search(r'"msg"\s*:\s*"([^"]*)"', x.get("text") or x.get("reply") or x.get("completion") or "")
                m = mm.group(1).strip() if mm else ""
                if not m: continue
                nm += 1; c = cat(m); C[c] += 1; ex[c][m.lower()] += 1; (W if w else L)[c] += 1
    print(f"\n{d}: decisions {n}, with msg {nm} ({nm/n:.1%}); team-coordination share {sum(C[c] for c in TEAM)/nm:.1%}")
    print(f"{'category':30s} {'share':>6s} {'in wins':>8s} {'in losses':>9s}  examples")
    for c, k in C.most_common():
        print(f"{c:30s} {k/nm:6.1%} {W[c]/max(1,sum(W.values())):8.1%} {L[c]/max(1,sum(L.values())):9.1%}  " +
              "; ".join(f'"{m}"' for m, _ in ex[c].most_common(3)))
