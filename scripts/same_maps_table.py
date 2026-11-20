"""Win rates of every model on the same fixed maps 200000-200063 (grouped vs split start), for slide 4.
usage: python scripts/same_maps_table.py > report/slides_src/same_maps.json"""
import json, glob, re, os
R = re.compile(r"a(\d), a (\w+) at \(([-\d.]+), ([-\d.]+)\)")
MAPS = range(200000, 200064)


def f64(run):
    """mappo.py writes eval_episodes.jsonl, train.py --mode eval writes episodes.jsonl."""
    W = {}
    f = f"runs/{run}/eval_episodes.jsonl"
    for l in open(f if os.path.exists(f) else f"runs/{run}/episodes.jsonl"):
        e = json.loads(l)
        if e.get("team", "A") == "A":
            W[e["scenario"]] = bool(e["info"].get("battle_won"))
    return W


def api(*pats):
    """Wins per scenario from play.py logs. A game with any API failure (e.g. 402 out of credit) is dropped;
    a clean single-game replay (runs/<pat>/s<scenario>.jsonl) replaces it."""
    W, clean = {}, {}
    for pat in pats:
        for f in glob.glob(f"runs/{pat}/w*.jsonl"):
            bad = 0
            for l in open(f):
                d = json.loads(l)
                if not d.get("summary"):
                    bad += sum(x["reply"].startswith("ERROR") for x in d["team_a"])
                    continue
                if d.get("scenario") is not None and not bad:
                    W[d["scenario"]] = bool(d["won"])
                bad = 0
        for f in glob.glob(f"runs/{pat}/s*.jsonl"):
            bad = 0
            for l in open(f):
                d = json.loads(l)
                if not d.get("summary"):
                    bad += sum(x["reply"].startswith("ERROR") for x in d["team_a"])
                elif not bad:
                    clean[d["scenario"]] = bool(d["won"])
    W.update(clean)
    return W


grouped = set()   # start type is fixed by the seed: our 5 units around the map centre
for l in open("runs/f64_sft/eval_episodes.jsonl"):
    e = json.loads(l)
    pos = [R.search(d["prompt"]) for d in e["decisions"] if int(d["t"]) == 0]
    if all(abs(float(m[3]) - 16) < 2.5 and abs(float(m[4]) - 16) < 2.5 for m in pos):
        grouped.add(e["scenario"])
models = [("Gemma 4 31B (teacher)", api("evalonly_31b_fixed40", "evalonly_31b_fixed64rest")),
          ("GPT-5.6 Luna", api("evalonly_luna_fixed40", "evalonly_luna_fixed64rest")),
          ("Gemini 3.5 Flash-Lite", api("evalonly_gflashlite_test", "evalonly_gflashlite_fixed40", "evalonly_gflashlite_fixed40b",
                                        "evalonly_gflashlite_fixed64rest", "evalonly_gflashlite_fix8")),
          ("4B MAPPO (best RL so far)", f64("f64_mappo")),
          ("4B SFT on 31B games", f64("f64_sft"))]
import os
for name, run in [("4B GRPO self-play", "f64_grpo"), ("4B GRPO vs built-in AI", "f64_grpo_ai"),
                  ("4B untrained", "f64_untrained"), ("4B untrained, no history", "f64_untrained_nohist")]:
    if glob.glob(f"runs/{run}/*episodes.jsonl"):
        W = f64(run)
        if len([k for k in MAPS if k in W]) == len(MAPS):   # only complete runs
            models.append((name, W))
rows = []
for name, W in models:
    ks = [k for k in MAPS if k in W]
    g = [k for k in ks if k in grouped]; s = [k for k in ks if k not in grouped]
    rows.append(dict(model=name, n=len(ks), wins=sum(W[k] for k in ks), win=round(100 * sum(W[k] for k in ks) / len(ks), 1),
                     n_grouped=len(g), grouped=round(100 * sum(W[k] for k in g) / max(len(g), 1), 1),
                     n_split=len(s), split=round(100 * sum(W[k] for k in s) / max(len(s), 1), 1)))
ref = dict(models)["4B SFT on 31B games"]
for r, (name, W) in zip(rows, models):
    ks = [k for k in MAPS if k in W and k in ref]
    r["only_this"] = sum(W[k] and not ref[k] for k in ks); r["only_sft"] = sum(ref[k] and not W[k] for k in ks)
print(json.dumps(dict(n_grouped=len(grouped), n_split=64 - len(grouped), rows=rows), indent=1))
