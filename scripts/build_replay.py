"""Turn one logged SMAC episode (JSONL from smac_llm.run) into a self-contained replay page.

usage: python scripts/build_replay.py runs/demo/luna_p5.jsonl --model "GPT-5.6 Luna" \
           --out artifacts/replay/index.html [--episode 0]
"""
import argparse
import json
import re
from pathlib import Path

MOVES = {0: "no_op", 1: "stop", 2: "move_north", 3: "move_south", 4: "move_east", 5: "move_west"}
N_NO_ATTACK = 6  # SMACv2 without conic field of view


def action_name(a: int) -> str:
    return MOVES.get(a, f"attack_e{a - N_NO_ATTACK}")


def note_of(reply: str) -> str:
    m = re.search(r"\{.*\}", reply or "", re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("note", ""))[:120]
        except json.JSONDecodeError:
            pass
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--model", required=True)
    p.add_argument("--matchup", default="5 Protoss vs 5 Protoss (SMACv2, built-in AI difficulty 7)")
    p.add_argument("--template", default="scripts/replay_template.html")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    frames, summary = [], None
    for line in open(args.log):
        d = json.loads(line)
        if d.get("episode") != args.episode:
            continue
        if d.get("summary"):
            summary = d
            continue
        acts = {}
        for r in d["agents"]:
            fallback = (r["reply"] or "").startswith("ERROR")
            acts[str(r["agent"])] = {"act": action_name(r["action"]), "note": note_of(r["reply"]),
                                     "valid": r["valid"], "src": "fallback" if fallback else "model",
                                     "obs": r["prompt"], "reply": r["reply"][:400]}
        frames.append({"t": d["step"], "r": round(d["reward"], 3), "u": d["units"], "a": acts})
    orders = [o for f in frames for o in f["a"].values()]
    by_model = [o for o in orders if o["src"] == "model"]
    first_fb = next((f["t"] for f in frames if any(o["src"] == "fallback" for o in f["a"].values())), None)
    stats = {"orders": len(orders), "model_orders": len(by_model),
             "fallback_orders": len(orders) - len(by_model),
             "model_illegal": sum(not o["valid"] for o in by_model), "first_fallback_step": first_fb}
    data = {"model": args.model, "matchup": args.matchup, "summary": summary, "stats": stats,
            "frames": frames}
    html = Path(args.template).read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}: {len(frames)} frames, summary={summary}")


if __name__ == "__main__":
    main()
