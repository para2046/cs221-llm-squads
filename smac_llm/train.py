"""Train Gemma (LoRA) to play SMACv2: imitation of a teacher, then online self-play RL.

Modes
  bc      Learn from logged teacher games (JSONL from smac_llm.run / scripts/collect.sh).
          --weighting filter : behaviour cloning on decisions from the best episodes only
          --weighting awr    : advantage-weighted regression (offline RL): every decision is
                               weighted by exp(A/beta), A = team return-to-go minus a per-step
                               baseline, so good decisions from losing games still count.
  online  Real-time RL. The model controls all units (shared weights, decentralised execution)
          against whatever opponent the env server runs (built-in AI, scripted, or another LLM).
          Each iteration plays a batch of live episodes, computes per-decision advantages from
          the shared team reward, and updates the LoRA weights. Objectives (--algo):
            ppo   per-step baseline (batch mean of return-to-go at that step), sequence-level
                  ratio, symmetric clip, k3 KL to the frozen base model
            grpo  critic-free group baseline: G rollouts of the SAME scenario (unit mix and
                  start positions); advantage = (return - group mean) / group std
            dapo  grpo + decoupled clip (clip-higher), dynamic sampling (drop groups whose
                  rollouts all scored the same), token-level loss, no KL by default
            gspo  grpo with a length-normalised sequence ratio (sequence-level clipping)
          Teacher guidance during RL (optional, combinable):
            --teacher-data/--teacher-coef  off-policy: AWR-weighted NLL on logged teacher
                  decisions added to every update (mixes imitation into on-policy RL)
            --distill-model/--distill-coef on-policy distillation (DAgger): the teacher labels
                  a sample of the states the STUDENT visited this iteration; NLL toward its action
  eval    Play only (no gradient), report win rate / return / illegal-order rate.

The environment runs on a CPU node (smac_llm.env_server); this process needs one GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from .text_env import SYSTEM_PROMPT


# ------------------------------------------------------------------------------- model utils
def load_model(name: str, adapter: str | None, lora_r: int, train: bool):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return tok, model


SYSTEM = {"text": SYSTEM_PROMPT}  # replaced by the env server's system prompt in online/eval mode


def chat_prefix(tok, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM["text"]}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def canonical_reply(reply: str, keep_say: bool = False) -> str | None:
    """Training target: the bare JSON action (drops teacher notes/reasoning); keeps the "say"
    message when the message channel is on."""
    m = re.search(r"\{.*?\}", reply or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        a = d.get("action")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not a:
        return None
    out = {"action": a}
    if keep_say and isinstance(d.get("say"), str) and d["say"].strip():
        out["say"] = " ".join(d["say"].split()[:12])
    if keep_say and isinstance(d.get("msg"), str) and d["msg"].strip():   # unified C2 channel
        out["msg"] = " ".join(d["msg"].split()[:8])
    return json.dumps(out)


_END = {}


def end_of_turn(tok) -> str:
    """The text the chat template puts right after an assistant reply (Gemma 4: '<turn|>').
    The model stops generating on this marker, not on <eos>, so training targets must end with it."""
    if id(tok) not in _END:
        s = tok.apply_chat_template([{"role": "user", "content": "u"},
                                     {"role": "assistant", "content": "XQXQ"}], tokenize=False)
        _END[id(tok)] = s.split("XQXQ", 1)[1].rstrip()
    return _END[id(tok)]


def seq_logprobs(model, tok, prefixes: list[str], completions: list[str], max_len=2048,
                 per_token=False):
    """Sum of token log-probs of each completion (plus end-of-turn) given its prefix: (B,) tensor.
    per_token=True returns (token log-probs (B,K), mask (B,K)) instead."""
    end = end_of_turn(tok)
    ids, labels = [], []
    for p, c in zip(prefixes, completions):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ci = tok(c + end, add_special_tokens=False)["input_ids"]
        x = (pi + ci)[-max_len:]
        lab = ([-100] * len(pi) + ci)[-max_len:]
        ids.append(x)
        labels.append(lab)
    L = max(map(len, ids))
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
    att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
    lab = torch.tensor([[-100] * (L - len(y)) + y for y in labels], device="cuda")
    # Left padding puts every completion at the end, so only the last K positions need logits
    # (full-vocabulary logits for whole prompts do not fit in memory: Gemma's vocab is 262k).
    K = max(sum(v != -100 for v in y) for y in labels) + 1
    # Note: do not pass explicit position_ids; for Gemma 4 the model derives them from the mask,
    # and passing cumsum positions increased padding sensitivity (0.19 vs 0.06 nats) in a test.
    logits = model(input_ids=inp, attention_mask=att, logits_to_keep=K).logits[:, :-1].float()
    tgt = lab[:, L - K + 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (tgt != -100).float()
    if per_token:
        return lp * mask, mask
    return (lp * mask).sum(-1)


GEN = {"max_new_tokens": None, "bs": int(os.environ.get("GEN_BS", "64"))}


@torch.no_grad()
def generate(model, tok, prompts: list[str], temperature: float, max_new_tokens=24, bs=64):
    """Sampled replies. Chunk size starts at GEN_BS and halves (for the rest of the run) on CUDA OOM."""
    max_new_tokens = GEN["max_new_tokens"] or max_new_tokens
    outs, i = [], 0
    while i < len(prompts):
        b = min(bs, GEN["bs"])
        chunk = [chat_prefix(tok, p) for p in prompts[i:i + b]]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                               temperature=max(temperature, 1e-5), top_p=1.0,
                               pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            enc = g = None
            torch.cuda.empty_cache()
            if b <= 1:
                raise
            GEN["bs"] = max(1, b // 2)
            print(f"generate: CUDA OOM at chunk {b}, now {GEN['bs']}", flush=True)
            continue
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        i += b
    return [o.strip() for o in outs]


# --------------------------------------------------------------------------------- bc mode
def _phi_v2(units):
    """Reward-v2 potential from a logged snapshot: 0.5*H + 0.25*L (team A's view)."""
    al, en = units["allies"], units["enemies"]
    hp = lambda us: sum(u["hp"] + u["sh"] for u in us)
    mx = lambda us: sum(u["hp_max"] + u["sh_max"] for u in us) or 1.0
    alive = lambda us: sum(u["hp"] > 0 for u in us)
    n = max(len(al), 1)
    return 0.5 * (hp(al) / mx(al) - hp(en) / mx(en)) + 0.25 * (alive(al) - alive(en)) / n


TEACHER_REWARD = ["smac"]
DAGGER_POOL = []  # --teacher-reward v2: recompute team reward v2 from logged unit states


def load_teacher(paths: list[str], gamma: float, keep_say: bool = False):
    """Decisions from logged games with their team return-to-go and episode outcome."""
    eps = {}
    for pat in paths:
        for f in sorted(glob.glob(pat if pat.endswith(".jsonl") else f"{pat}/*.jsonl")):
            for line in open(f):
                d = json.loads(line)
                key = (f, d["episode"])
                if d.get("summary"):
                    eps.setdefault(key, {"steps": []})["summary"] = d
                else:
                    eps.setdefault(key, {"steps": []})["steps"].append(d)
    data = []
    for key, ep in eps.items():
        if "summary" not in ep:
            continue
        steps = sorted(ep["steps"], key=lambda s: s["step"])
        if TEACHER_REWARD[0] == "v2":   # r_t = Phi(s_t+1) - Phi(s_t), outcome +-2 (draw 0) at the end
            sm = ep["summary"]
            phis = [_phi_v2(s["units"]) for s in steps]
            n = max(len(steps[0]["units"]["allies"]), 1)
            dead_a, dead_e = sm.get("dead_allies") or 0, sm.get("dead_enemies") or 0
            last = steps[-1]["units"]
            fin = dict(allies=[dict(u, hp=u["hp"] if i >= dead_a else 0, sh=u["sh"] if i >= dead_a else 0)
                               for i, u in enumerate(sorted(last["allies"], key=lambda u: u["hp"] + u["sh"]))],
                       enemies=[dict(u, hp=u["hp"] if i >= dead_e else 0, sh=u["sh"] if i >= dead_e else 0)
                                for i, u in enumerate(sorted(last["enemies"], key=lambda u: u["hp"] + u["sh"]))])
            phis.append(_phi_v2(fin))   # final state approximated: the weakest units counted as dead
            win = sm.get("winner") == "A" or sm.get("won")
            lost = sm.get("winner") == "B"
            rew = [phis[i + 1] - phis[i] for i in range(len(steps))]
            rew[-1] += 2.0 if win else (-2.0 if lost else 0.0)
        else:
            rew = [s["reward"] for s in steps]
        rtg, g = [0.0] * len(steps), 0.0
        for i in range(len(steps) - 1, -1, -1):
            g = rew[i] + gamma * g
            rtg[i] = g
        for i, s in enumerate(steps):
            for r in (s.get("agents") or s.get("team_a") or []):
                tgt = canonical_reply(r["reply"], keep_say)
                if tgt and r["valid"]:
                    data.append({"prompt": r["prompt"], "target": tgt, "rtg": rtg[i],
                                 "t": s["step"], "ep_return": ep["summary"]["return"],
                                 "won": ep["summary"]["won"], "ep": str(key)})
    return data


def run_bc(args, tok, model):
    if args.env_url:   # same system prompt and channel as the env the student will play in
        info = EnvClient(args.env_url, open(args.env_token_file).read().strip()).post("/info", {})
        SYSTEM.update(text=info.get("system", SYSTEM_PROMPT), channel=info.get("channel", "none"))
        print(f"bc: system prompt from env ({SYSTEM['channel']}), {len(SYSTEM['text'])} chars", flush=True)
    data = load_teacher(args.data, args.gamma, keep_say=SYSTEM.get("channel", "none") != "none")
    print(f"loaded {len(data)} teacher decisions from {len(set(d['ep'] for d in data))} episodes")
    if args.weighting == "filter":
        rets = sorted({(d["ep"], d["ep_return"]) for d in data}, key=lambda x: -x[1])
        keep = {e for e, _ in rets[: max(1, int(len(rets) * args.keep_frac))]}
        data = [dict(d, w=1.0) for d in data if d["ep"] in keep]
    else:  # awr: baseline = mean return-to-go at the same step index
        by_t = {}
        for d in data:
            by_t.setdefault(d["t"], []).append(d["rtg"])
        base = {t: sum(v) / len(v) for t, v in by_t.items()}
        advs = [d["rtg"] - base[d["t"]] for d in data]
        sd = (sum(a * a for a in advs) / len(advs)) ** 0.5 + 1e-6
        data = [dict(d, w=min(math.exp(a / sd / args.awr_beta), 20.0)) for d, a in zip(data, advs)]
    if args.min_weight > 0:
        data = [d for d in data if d["w"] >= args.min_weight]
    print(f"training on {len(data)} decisions, mean weight {sum(d['w'] for d in data)/len(data):.3f}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log = open(Path(args.out) / "train_log.jsonl", "a")
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch):
            b = data[i:i + args.batch]
            try:
                lp = seq_logprobs(model, tok, [chat_prefix(tok, d["prompt"]) for d in b],
                                  [d["target"] for d in b])
                w = torch.tensor([d["w"] for d in b], device="cuda")
                loss = -(w * lp).sum() / w.sum()
                (loss / args.grad_accum).backward()
            except torch.OutOfMemoryError:   # rare very long prompt: skip it
                lp = loss = None
                torch.cuda.empty_cache()
                print("bc: CUDA OOM, skipped a minibatch", flush=True)
                continue
            step += 1
            if step % args.grad_accum == 0 or i + args.batch >= len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if args.save_steps and step % args.save_steps == 0:
                model.save_pretrained(Path(args.out) / "adapter")
            if step % 10 == 0:
                rec = {"mode": "bc", "epoch": epoch, "step": step, "loss": round(loss.item(), 4)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
    model.save_pretrained(Path(args.out) / "adapter")
    print(f"saved adapter to {Path(args.out) / 'adapter'}")


# ------------------------------------------------------------------------------ online mode
class EnvClient:
    def __init__(self, url: str, token: str):
        import threading
        self.local = threading.local()  # one HTTP session per thread
        self.url, self.h = url.rstrip("/"), {"Authorization": f"Bearer {token}"}
        self.n = self.post("/info", {})["n_envs"]

    def post(self, path, data, retries=5):
        import requests
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        for k in range(retries):
            try:
                r = self.local.s.post(self.url + path, json=data, headers=self.h, timeout=600)
                r.raise_for_status()
                return r.json()
            except requests.ConnectionError:
                if k == retries - 1 or path == "/step":  # never replay a step: it is not idempotent
                    raise
                time.sleep(1 + k)
                self.local.s = requests.Session()


def play_batch(client, model, tok, n_episodes: int, temperature: float, scenarios=None):
    """Play n episodes concurrently (one per env slot, round-robin). scenarios[k] (optional)
    makes episode k replay a fixed scenario (for group baselines). Returns episode records."""
    pool = ThreadPoolExecutor(client.n)
    episodes = []
    done_n = 0
    while done_n < n_episodes:
        k = min(client.n, n_episodes - done_n)
        scen = [scenarios[done_n + i] if scenarios else None for i in range(k)]
        states = list(pool.map(lambda i: client.post("/reset", {"env": i, "scenario": scen[i]}), range(k)))
        for st in states:
            if "error" in st:
                raise RuntimeError(st["error"] + "\n" + st.get("trace", ""))
        eps = [{"decisions": [], "rewards": [], "smac": [], "info": {}, "scenario": scen[i], "team_b": [],
                "team": "A"} for i in range(k)]
        selfplay = "prompts_b" in states[0]
        eps_b = [{"decisions": [], "rewards": [], "smac": [], "info": {}, "team": "B", "team_b": [],
                  "scenario": f"{scen[i]}B" if scen[i] is not None else None} for i in range(k)] if selfplay else []
        live = set(range(k))
        while live:
            batch = [(i, "A", a, p) for i in sorted(live) for a, p in states[i]["prompts"].items()]
            if selfplay:  # the same (current) model plays both teams
                batch += [(i, "B", a, p) for i in sorted(live) for a, p in states[i]["prompts_b"].items()]
            if OPP_BASE["team"] and selfplay:  # head-to-head: one team plays with the LoRA disabled (= untrained model)
                idx_base = [j for j, x in enumerate(batch) if x[1] == OPP_BASE["team"]]
                idx_tr = [j for j, x in enumerate(batch) if x[1] != OPP_BASE["team"]]
                replies = [None] * len(batch)
                for j, r in zip(idx_tr, generate(model, tok, [batch[j][3] for j in idx_tr], temperature)):
                    replies[j] = r
                with model.disable_adapter():
                    for j, r in zip(idx_base, generate(model, tok, [batch[j][3] for j in idx_base], temperature)):
                        replies[j] = r
            else:
                replies = generate(model, tok, [p for *_, p in batch], temperature)
            per_env, per_env_b = {}, {}
            for (i, team, a, p), rep in zip(batch, replies):
                (per_env if team == "A" else per_env_b).setdefault(i, {})[a] = rep
                ep = eps[i] if team == "A" else eps_b[i]
                ep["decisions"].append({"t": len(ep["rewards"]), "agent": a, "prompt": p, "reply": rep})
            res = list(pool.map(lambda i: (i, client.post("/step", {"env": i, "replies": per_env.get(i, {}),
                                                                     "replies_b": per_env_b.get(i, {})})),
                                sorted(live)))
            for i, st in res:
                if "error" in st:
                    raise RuntimeError(st["error"] + "\n" + st.get("trace", ""))
                eps[i]["rewards"].append(st["reward"])
                eps[i]["smac"].append(st["info"].get("reward_smac", st["reward"]))
                t = len(eps[i]["rewards"]) - 1
                for d in eps[i]["decisions"]:
                    if d["t"] == t:
                        d["valid"] = st.get("valid", {}).get(d["agent"], True)
                        d["own"] = st.get("reward_unit", {}).get(d["agent"], 0.0)
                if selfplay:
                    eb = eps_b[i]
                    eb["rewards"].append(st.get("reward_b", 0.0))
                    eb["smac"].append(st["info"].get("reward_smac_b", st.get("reward_b", 0.0)))
                    for d in eb["decisions"]:
                        if d["t"] == t:
                            d["valid"] = st.get("valid_b", {}).get(d["agent"], True)
                            d["own"] = st.get("reward_unit_b", {}).get(d["agent"], 0.0)
                elif st.get("team_b"):
                    eps[i]["team_b"].append({"t": t, "agents": st["team_b"]})
                states[i] = st
                if st["done"]:
                    eps[i]["info"] = st["info"]
                    if selfplay:
                        inf = st["info"]
                        eps_b[i]["info"] = {"battle_won": inf.get("winner") == "B", "winner": inf.get("winner"),
                                            "dead_enemies": inf.get("dead_allies", 0),
                                            "reward_parts": inf.get("reward_parts_b", {})}
                    live.discard(i)
        episodes += eps + eps_b
        done_n += k
    return episodes


def _rtg(rewards, gamma):
    out, g = [0.0] * len(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def _own_rtg(ep, gamma):
    """Per-unit shaped reward (reward_unit from the server), as return-to-go for that unit's own
    decisions: {id(decision): G_own}. Zero for the plain SMAC reward."""
    by_agent = {}
    for d in ep["decisions"]:
        by_agent.setdefault(d["agent"], []).append(d)
    out = {}
    for ds in by_agent.values():
        ds.sort(key=lambda d: d["t"])
        g = 0.0
        for d in reversed(ds):
            g = d.get("own", 0.0) + gamma * g
            out[id(d)] = g
    return out


def advantages(episodes, gamma: float, invalid_penalty: float):
    """PPO-style: team return-to-go per step, baseline = batch mean at the same step, normalized."""
    rows = []
    for ep in episodes:
        rtg, own = _rtg(ep["rewards"], gamma), _own_rtg(ep, gamma)
        for d in ep["decisions"]:
            rows.append((d, rtg[d["t"]] + own.get(id(d), 0.0)))
    by_t = {}
    for d, g in rows:
        by_t.setdefault(d["t"], []).append(g)
    base = {t: sum(v) / len(v) for t, v in by_t.items()}
    adv = [g - base[d["t"]] - (invalid_penalty if not d.get("valid", True) else 0.0) for d, g in rows]
    sd = (sum(a * a for a in adv) / max(len(adv), 1)) ** 0.5 + 1e-6
    return [(d, a / sd) for (d, _), a in zip(rows, adv)]


OPP_BASE = {"team": None}  # --base-team A|B: that team plays the untrained base model (eval only)
ADV_SD_FLOOR = [0.0]  # --adv-sd-floor: lower bound on the group std (reward v2 uses 0.5)


def group_advantages(episodes, gamma: float, invalid_penalty: float, credit: str,
                     min_group_std: float = 0.0):
    """GRPO-style advantages: rollouts of the same scenario form a group (no critic).
    credit='episode': A = (R - mean_g R) / std_g R for every decision of the episode.
    credit='rtg':     A_t = (G_t - mean_g G_t) / std_g R, G_t = return-to-go at step t (a
                      rollout that already ended contributes G_t = 0).
    Groups whose returns are all (nearly) equal carry no signal; with min_group_std > 0 they are
    dropped (DAPO's dynamic sampling). Returns (samples, n_groups_kept, n_groups)."""
    groups = {}
    for ep in episodes:
        groups.setdefault(ep["scenario"], []).append(ep)
    out, kept = [], 0
    for g in groups.values():
        R = [sum(e["rewards"]) for e in g]
        mu = sum(R) / len(R)
        sd = (sum((r - mu) ** 2 for r in R) / max(len(R) - 1, 1)) ** 0.5
        if len(g) < 2 or sd <= min_group_std:
            continue
        kept += 1
        rtgs = [_rtg(e["rewards"], gamma) for e in g]
        for e, rtg, r in zip(g, rtgs, R):
            own = _own_rtg(e, gamma)
            for d in e["decisions"]:
                if credit == "episode":
                    a = (r - mu) / max(sd + 1e-6, ADV_SD_FLOOR[0])
                else:
                    t = d["t"]
                    base = sum(x[t] if t < len(x) else 0.0 for x in rtgs) / len(rtgs)
                    a = (rtg[t] + own.get(id(d), 0.0) - base) / max(sd + 1e-6, ADV_SD_FLOOR[0])
                if not d.get("valid", True):
                    a -= invalid_penalty
                out.append((d, a))
    return out, kept, len(groups)


def ep_stats(episodes):
    n = len(episodes)
    dec = [d for e in episodes for d in e["decisions"]]
    st = {"episodes": n,
          "win_rate": round(sum(bool(e["info"].get("battle_won")) for e in episodes) / n, 3),
          "return": round(sum(sum(e.get("smac") or e["rewards"]) for e in episodes) / n, 3),
          "dead_enemies": round(sum(e["info"].get("dead_enemies", 0) for e in episodes) / n, 2),
          "invalid_rate": round(sum(not d.get("valid", True) for d in dec) / max(len(dec), 1), 3)}
    if any("reward_parts" in e["info"] for e in episodes):  # shaped training reward
        st["shaped_return"] = round(sum(sum(e["rewards"]) for e in episodes) / n, 3)
        for k in ("hp", "kill", "death", "outcome", "focus"):
            st["r_" + k] = round(sum(e["info"].get("reward_parts", {}).get(k, 0) for e in episodes) / n, 3)
    if any("winner" in e["info"] for e in episodes):  # model-vs-model
        st["loss_rate"] = round(sum(e["info"].get("winner") == "B" for e in episodes) / n, 3)
        st["return_b"] = round(sum(e["info"].get("return_b", 0) for e in episodes) / n, 3)
    if SYSTEM.get("channel", "none") != "none":
        st["say_rate"] = round(sum('"say"' in (d["reply"] or "") for d in dec) / max(len(dec), 1), 3)
    return st


# ----------------------------------------------------------------- teacher guidance helpers
def teacher_pool(args):
    """Logged teacher decisions with AWR weights, for the off-policy auxiliary loss."""
    data = load_teacher(args.teacher_data, args.gamma, keep_say=SYSTEM.get("channel", "none") != "none")
    by_t = {}
    for d in data:
        by_t.setdefault(d["t"], []).append(d["rtg"])
    base = {t: sum(v) / len(v) for t, v in by_t.items()}
    advs = [d["rtg"] - base[d["t"]] for d in data]
    sd = (sum(a * a for a in advs) / max(len(advs), 1)) ** 0.5 + 1e-6
    for d, a in zip(data, advs):
        d["w"] = min(math.exp(a / sd / args.awr_beta), 20.0)
    print(f"teacher pool: {len(data)} decisions", flush=True)
    return data


def distill_labels(args, decisions):
    """On-policy distillation / DAgger: ask the teacher what it would do in states the student
    actually visited. Returns [(prompt, canonical teacher reply, student reply)]."""
    from openai import OpenAI
    base_url, model_name = args.distill_model.rsplit(":", 1)
    client = OpenAI(base_url=base_url, api_key=os.environ.get(args.distill_key_env or "", "EMPTY"),
                    timeout=120)
    keep_say = SYSTEM.get("channel", "none") != "none"

    def one(d):
        try:
            r = client.chat.completions.create(
                model=model_name, temperature=0.0, max_tokens=96,
                messages=[{"role": "system", "content": SYSTEM["text"]},
                          {"role": "user", "content": d["prompt"]}])
            return d["prompt"], canonical_reply(r.choices[0].message.content or "", keep_say), d["reply"]
        except Exception:
            return d["prompt"], None, d["reply"]

    with ThreadPoolExecutor(32) as ex:
        out = [x for x in ex.map(one, decisions) if x[1]]
    return out


def weighted_nll(model, tok, prompts, targets, weights):
    lp = seq_logprobs(model, tok, [chat_prefix(tok, p) for p in prompts], targets)
    w = torch.tensor(weights, device="cuda", dtype=lp.dtype)
    return -(w * lp).sum() / w.sum()


ALGO_DEFAULTS = {  # clip_low, clip_high, kl
    "ppo": (0.2, 0.2, 0.05), "grpo": (0.2, 0.2, 0.04), "dapo": (0.2, 0.28, 0.0),
    "gspo": (0.003, 0.004, 0.0)}


def eval_fixed(args, model, tok, it):
    """Evaluate the current weights on a second env server (e.g. vs the built-in AI)."""
    client = EnvClient(args.eval_env_url, open(args.env_token_file).read().strip())
    info = client.post("/info", {})
    saved, saved_gen = dict(SYSTEM), dict(GEN)
    SYSTEM.update(text=info.get("system", SYSTEM_PROMPT), channel=info.get("channel", "none"))
    model.eval()
    model.config.use_cache = True
    eps = play_batch(client, model, tok, args.eval_episodes, 0.7)
    SYSTEM.clear(); SYSTEM.update(saved); GEN.update(saved_gen)
    torch.cuda.empty_cache()
    st = {"mode": "eval_fixed", "iter": it, "opponent": info.get("opponent"), **ep_stats(eps)}
    print("eval_fixed", json.dumps(st), flush=True)
    with open(Path(args.out) / "eval_fixed.jsonl", "a") as f:
        f.write(json.dumps(st) + "\n")
    if args.save_episodes:
        with open(Path(args.out) / "eval_fixed_episodes.jsonl", "a") as f:
            for e in eps:
                f.write(json.dumps({"iter": it, **e}) + "\n")


def run_online(args, tok, model, train: bool):
    client = EnvClient(args.env_url, open(args.env_token_file).read().strip())
    info = client.post("/info", {})
    SYSTEM.update(text=info.get("system", SYSTEM_PROMPT), channel=info.get("channel", "none"))
    if SYSTEM["channel"] != "none":
        GEN["max_new_tokens"] = max(args.max_new_tokens, 48)
    else:
        GEN["max_new_tokens"] = args.max_new_tokens
    print("env:", json.dumps({k: v for k, v in info.items() if k != "system"}), flush=True)
    clip_lo, clip_hi, kl_coef = ALGO_DEFAULTS[args.algo]
    clip_lo = args.clip_low if args.clip_low is not None else clip_lo
    clip_hi = args.clip_high if args.clip_high is not None else clip_hi
    kl_coef = args.kl if args.kl is not None else kl_coef
    grouped = args.algo in ("grpo", "dapo", "gspo")
    tpool = teacher_pool(args) if train and args.teacher_data else []
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr) if train else None
    log = open(Path(args.out) / ("train_log.jsonl" if train else "eval_log.jsonl"), "a")
    rng = random.Random(args.seed + 7)
    sil_buf = []  # off-policy self-imitation: (prompt, reply) of winning episodes from past iterations
    if train and args.eval_env_url and args.eval_every > 0 and not os.environ.get("SKIP_EVAL0"):
        eval_fixed(args, model, tok, 0)
    for it in range(args.iters):
        t0 = time.time()
        model.eval()
        model.config.use_cache = True
        scen = None
        if grouped and train:
            G = args.group_size
            n_groups = max(1, args.episodes_per_iter // G)
            ids = [rng.randrange(1 << 30) for _ in range(n_groups)]
            scen = [s for s in ids for _ in range(G)]
        if not train and args.eval_scenarios is not None:  # fixed, shared start scenarios for paired evaluation
            scen = [args.eval_scenarios + it * args.episodes_per_iter + j for j in range(args.episodes_per_iter)]
        n_eps = len(scen) if scen else args.episodes_per_iter
        episodes = play_batch(client, model, tok, n_eps, args.temperature, scen)
        stats = ep_stats([e for e in episodes if e.get("team", "A") == "A"])
        rec = {"mode": "online" if train else "eval", "algo": args.algo, "iter": it, **stats,
               "mean_steps": round(sum(len(e["rewards"]) for e in episodes) / len(episodes), 1),
               "rollout_s": round(time.time() - t0, 1)}
        print("rollout", json.dumps(rec), flush=True)
        if train:
            if grouped:
                samples, kept, ng = group_advantages(
                    episodes, args.gamma, args.invalid_penalty, args.credit,
                    min_group_std=args.min_group_std if args.algo == "dapo" else -1.0)
                rec.update(groups_kept=kept, groups=ng)
            else:
                samples = advantages(episodes, args.gamma, args.invalid_penalty)
            if len(samples) > args.max_decisions:  # uniform subsample: unbiased, bounded update cost
                samples = random.sample(samples, args.max_decisions)
            distill = []
            if args.distill_model and args.distill_n > 0:
                dec = [d for e in episodes for d in e["decisions"]]
                distill = distill_labels(args, random.sample(dec, min(args.distill_n, len(dec))))
                rec["distill_labels"] = len(distill)
                # how often the student already picked the teacher's action in these states
                rec["teacher_agree"] = round(sum(
                    json.loads(t)["action"] == (json.loads(canonical_reply(r) or "{}").get("action"))
                    for _, t, r in distill) / max(len(distill), 1), 3)
                distill = [(p_, t) for p_, t, _ in distill]
                if args.distill_aggregate:   # DAgger: train on every label collected so far
                    DAGGER_POOL.extend(distill)
                    distill = list(DAGGER_POOL)
                    rec["dagger_pool"] = len(distill)
            model.train()
            model.config.use_cache = False
            losses, kls, clipfrac, aux = [], [], [], []
            if samples:
                token_level = args.algo == "dapo"
                with torch.no_grad():  # behaviour-policy and reference log-probs
                    old, ref = [], []
                    for i in range(0, len(samples), args.batch):
                        b = samples[i:i + args.batch]
                        pre = [chat_prefix(tok, d["prompt"]) for d, _ in b]
                        comp = [d["reply"] for d, _ in b]
                        if token_level:
                            o, m = seq_logprobs(model, tok, pre, comp, per_token=True)
                            old += [(o[j], m[j]) for j in range(len(b))]
                        else:
                            old.append(seq_logprobs(model, tok, pre, comp))
                        if kl_coef > 0:
                            with model.disable_adapter():
                                ref.append(seq_logprobs(model, tok, pre, comp))
                    if not token_level:
                        old = torch.cat(old)
                    ref = torch.cat(ref) if ref else None
                n_mb = 0
                for _ in range(args.ppo_epochs):
                    order = list(range(len(samples)))
                    random.shuffle(order)
                    for i in range(0, len(order), args.batch):
                        idx = order[i:i + args.batch]
                        b = [samples[j] for j in idx]
                        pre = [chat_prefix(tok, d["prompt"]) for d, _ in b]
                        comp = [d["reply"] for d, _ in b]
                        A = torch.tensor([a for _, a in b], device="cuda")
                        if token_level:  # DAPO: per-token ratios, loss averaged over all tokens
                            lp_t, m = seq_logprobs(model, tok, pre, comp, per_token=True)
                            K = lp_t.shape[1]
                            old_t = torch.stack([F.pad(old[j][0], (K - old[j][0].shape[0], 0))[-K:] for j in idx])
                            ratio = torch.exp((lp_t - old_t) * m)
                            obj = torch.min(ratio * A[:, None],
                                            ratio.clamp(1 - clip_lo, 1 + clip_hi) * A[:, None])
                            pg = -(obj * m).sum() / m.sum()
                            lp = (lp_t * m).sum(-1)
                            cf = ((((ratio - 1).abs() > clip_lo) & (m > 0)).float().sum() / m.sum()).item()
                        else:
                            if args.algo == "gspo":
                                lp_t, m = seq_logprobs(model, tok, pre, comp, per_token=True)
                                lp = lp_t.sum(-1)
                                ratio = torch.exp((lp - old[idx]) / m.sum(-1).clamp(min=1))
                            else:
                                lp = seq_logprobs(model, tok, pre, comp)
                                ratio = torch.exp(lp - old[idx])
                            pg = -torch.min(ratio * A, ratio.clamp(1 - clip_lo, 1 + clip_hi) * A).mean()
                            cf = (((ratio - 1).abs() > clip_lo).float().mean()).item()
                        loss = args.pg_coef * pg
                        if kl_coef > 0:
                            lr_ = ref[idx] - lp  # k3 KL estimator (per sequence)
                            kl = (torch.exp(lr_) - lr_ - 1).mean()
                            loss = loss + kl_coef * kl
                            kls.append(kl.item())
                        if tpool and args.teacher_coef > 0:  # off-policy teacher guidance
                            tb = random.sample(tpool, min(args.batch, len(tpool)))
                            ta = weighted_nll(model, tok, [d["prompt"] for d in tb], [d["target"] for d in tb],
                                              [d["w"] for d in tb])
                            loss = loss + args.teacher_coef * ta
                            aux.append(ta.item())
                        if sil_buf and args.sil_coef > 0:  # off-policy self-imitation of past wins
                            sb = random.sample(sil_buf, min(args.batch, len(sil_buf)))
                            sa = weighted_nll(model, tok, [p for p, _ in sb], [r for _, r in sb], [1.0] * len(sb))
                            loss = loss + args.sil_coef * sa
                            aux.append(sa.item())
                        if distill and args.distill_coef > 0:  # on-policy distillation
                            db = random.sample(distill, min(args.batch, len(distill)))
                            da = weighted_nll(model, tok, [p for p, _ in db], [t for _, t in db], [1.0] * len(db))
                            loss = loss + args.distill_coef * da
                            aux.append(da.item())
                        (loss / args.grad_accum).backward()
                        n_mb += 1
                        if n_mb % args.grad_accum == 0 or i + args.batch >= len(order):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                            opt.zero_grad()
                        losses.append(loss.item())
                        clipfrac.append(cf)
            rec.update(loss=round(sum(losses) / max(len(losses), 1), 4),
                       kl=round(sum(kls) / len(kls), 4) if kls else None,
                       clipfrac=round(sum(clipfrac) / max(len(clipfrac), 1), 3), n_decisions=len(samples),
                       aux_loss=round(sum(aux) / len(aux), 4) if aux else None)
            if args.sil_coef > 0:  # add this iteration's winning episodes (valid decisions) to the buffer
                for e in episodes:
                    if e["info"].get("battle_won"):
                        sil_buf += [(d["prompt"], d["reply"]) for d in e["decisions"] if d.get("valid", True)]
                sil_buf = sil_buf[-args.sil_size:]
                rec["sil_buffer"] = len(sil_buf)
            if (it + 1) % args.save_every == 0 or it == args.iters - 1:
                model.save_pretrained(Path(args.out) / f"adapter_it{it + 1}")
            if args.eval_env_url and args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                eval_fixed(args, model, tok, it + 1)
        rec["iter_s"] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if args.save_episodes:
            with open(Path(args.out) / "episodes.jsonl", "a") as f:
                for e in episodes:
                    f.write(json.dumps({"iter": it, **e}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bc", "online", "eval"], required=True)
    p.add_argument("--eval-env-url", default=None, help="second env server for periodic fixed-opponent eval")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--sil-coef", type=float, default=0.0, help="off-policy self-imitation of past winning episodes")
    p.add_argument("--sil-size", type=int, default=4096)
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", default=None, help="start from this LoRA adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    # bc
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--weighting", choices=["filter", "awr"], default="awr")
    p.add_argument("--keep-frac", type=float, default=0.5)
    p.add_argument("--awr-beta", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=2)
    # online / eval
    p.add_argument("--env-url", default=None)
    p.add_argument("--env-token-file", default=None)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--episodes-per-iter", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--algo", choices=["ppo", "grpo", "dapo", "gspo"], default="ppo")
    p.add_argument("--group-size", type=int, default=4, help="grpo/dapo/gspo: rollouts per scenario")
    p.add_argument("--credit", choices=["episode", "rtg"], default="rtg",
                   help="grouped algos: whole-episode advantage or per-step return-to-go")
    p.add_argument("--min-group-std", type=float, default=0.05, help="dapo dynamic sampling")
    p.add_argument("--clip-low", type=float, default=None, help="default depends on --algo")
    p.add_argument("--clip-high", type=float, default=None)
    p.add_argument("--clip", type=float, default=None, help="symmetric clip (sets both)")
    p.add_argument("--kl", type=float, default=None, help="KL coefficient; default depends on --algo")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--teacher-data", nargs="*", default=[], help="logged teacher games (off-policy aux)")
    p.add_argument("--teacher-coef", type=float, default=0.0)
    p.add_argument("--distill-model", default=None, help="<base_url>:<model> teacher for on-policy distillation")
    p.add_argument("--distill-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--distill-n", type=int, default=256, help="student states labelled per iteration")
    p.add_argument("--distill-coef", type=float, default=0.0)
    p.add_argument("--distill-aggregate", action="store_true", help="DAgger: train on all labels so far")
    p.add_argument("--pg-coef", type=float, default=1.0, help="0 = no policy-gradient term (pure DAgger)")
    p.add_argument("--teacher-reward", choices=["smac", "v2"], default="smac")
    p.add_argument("--min-weight", type=float, default=0.0, help="bc: drop decisions with AWR weight below this")
    p.add_argument("--save-steps", type=int, default=0, help="bc: overwrite the adapter every N minibatches")
    p.add_argument("--invalid-penalty", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--max-decisions", type=int, default=1024, help="decisions per PPO update")
    p.add_argument("--grad-accum", type=int, default=1, help="minibatches per optimizer step")
    p.add_argument("--save-episodes", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adv-sd-floor", type=float, default=0.0)
    p.add_argument("--eval-scenarios", type=int, default=None, help="eval mode: scenario ids start..start+iters*eps-1")
    p.add_argument("--base-team", choices=["A", "B"], default=None, help="eval vs external: this team uses the untrained base")
    args = p.parse_args()
    ADV_SD_FLOOR[0] = args.adv_sd_floor
    TEACHER_REWARD[0] = args.teacher_reward
    OPP_BASE["team"] = args.base_team
    if args.clip is not None:
        args.clip_low = args.clip_high = args.clip
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(Path(args.out) / "config.json", "w"), indent=2)
    tok, model = load_model(args.model, args.adapter, args.lora_r, train=args.mode != "eval")
    if args.mode == "bc":
        run_bc(args, tok, model)
    else:
        run_online(args, tok, model, train=args.mode == "online")


if __name__ == "__main__":
    main()
