"""MAPPO for the LLM squad: shared LoRA actor (Gemma 4 E4B) + centralised critic on the SMAC global state.

Setting: team A (5 units) = the LLM, each unit acts from its own text prompt (decentralised execution);
opponent = SMAC built-in AI (stationary). Training reward = server reward (reward v2 "shaped2", zero-sum
potential-based team reward r_t, plus per-unit credit r^u_t that sums to 0 over the team); it is never
shown to the agents. The critic sees the global state s_t (training only).

One iteration
  1. Rollout (on-policy, temperature 1.0): N episodes on random scenarios. Per joint step t we store the
     global state s_t, team reward r_t, per-unit reward r^u_t, done; per decision (t, unit u) the prompt,
     the SAMPLED completion token ids, their log-prob at sampling time, the decoded reply, validity.
  2. Critic V(s, onehot(u)) (MLP) -> per-agent GAE on the joint timeline:
        delta_t^u = r_t + r_t^u - pen_t^u + gamma * V(s_{t+1}, u) * (1 - terminal_{t+1}) - V(s_t, u)
        A_t^u     = sum_k (gamma*lambda)^k delta_{t+k}^u,   R_t^u = A_t^u + V(s_t, u)
     The episode end (win / loss / 200-step draw) is terminal (the time limit is part of the game rules).
     Steps after unit u died stay in the recursion (team reward keeps flowing) but carry no actor loss.
  3. Actor: token-level PPO-clip on the sampled token ids, advantage normalised over the batch, k3 KL to
     the frozen SFT adapter ("ref"); old log-probs recomputed with the pre-update weights (the sampling-time
     values are logged to check the match).
  4. Critic: regression to R_t^u.
  Diagnostics per iteration: policy loss, grad norm, approx-KL(old||new), KL to SFT, clip fraction,
  advantage percentiles, critic explained variance (before its update), LoRA parameter change vs init,
  sampling-vs-recomputed log-prob gap.
Every --eval-every iterations: fixed scenarios (100000.., temperature 0.7) vs the built-in AI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn

from .text_env import SYSTEM_PROMPT
from .train import EnvClient, GEN, SYSTEM, chat_prefix, load_model


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
class SysPrompt(str):
    """A prompt that carries its own system prompt (games vs an API model use the model-vs-model system
    prompt, without the built-in-AI behaviour section). Plain str prompts use the server's SYSTEM."""
    def __new__(cls, text, system):
        o = super().__new__(cls, text)
        o.system = system
        return o


def _prefix(tok, p):
    sysp = getattr(p, "system", None)
    if sysp is None:
        return chat_prefix(tok, p)
    msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": str(p)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_ids(model, tok, prompts, temperature, max_new_tokens, eval_sampling=False):
    """Sample replies; returns [(completion_ids, per_token_logprobs_at_sampling, text)].
    Training: pure ancestral sampling from pi (top_k=0, top_p=1; Gemma's generation_config defaults
    top_k=64/top_p=0.95 are overridden), so the stored log-probs are exactly the behaviour policy.
    eval_sampling=True reproduces train.py's evaluation (top_k=64 default, top_p=1.0) for comparability."""
    out, i = [], 0
    stop = set(x for x in [tok.eos_token_id, tok.pad_token_id] if x is not None)
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        stop |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    while i < len(prompts):
        b = min(len(prompts) - i, GEN["bs"])
        chunk = [_prefix(tok, p) for p in prompts[i:i + b]]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                               temperature=temperature, top_p=1.0, top_k=64 if eval_sampling else 0,
                               repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
                               return_dict_in_generate=True, output_scores=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b <= 1:
                raise
            GEN["bs"] = max(1, b // 2)
            print(f"generate: OOM, chunk -> {GEN['bs']}", flush=True)
            continue
        seq = g.sequences[:, enc["input_ids"].shape[1]:]
        # log-prob of each sampled token under the sampling distribution (temperature-scaled scores)
        tlp = torch.stack([torch.log_softmax(s.float(), -1).gather(-1, seq[:, k:k + 1]).squeeze(-1)
                           for k, s in enumerate(g.scores)], 1).cpu()
        seq = seq.cpu().tolist()
        for j in range(len(seq)):
            ids, lp = [], []
            for k, t in enumerate(seq[j]):
                lp.append(float(tlp[j, k]))
                ids.append(t)
                if t in stop:
                    break
            out.append((ids, lp, tok.decode(ids, skip_special_tokens=True).strip()))
        del g, tlp
        i += b
    return out


def token_logprobs(model, tok, prompts, comp_ids, max_len=4096):
    """Per-token log-probs of given completion ids: (B, K) lp and mask (right-aligned, left padding)."""
    ids, ncomp = [], []
    for p, c in zip(prompts, comp_ids):
        pi = tok(_prefix(tok, p), add_special_tokens=False)["input_ids"]
        x = (pi + list(c))[-max_len:]
        ids.append(x)
        ncomp.append(len(c))
    L, pad = max(map(len, ids)), tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
    att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
    K = max(ncomp)
    logits = model(input_ids=inp, attention_mask=att, logits_to_keep=K + 1).logits[:, :-1].float()
    tgt = inp[:, L - K:]
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(lp)
    for j, n in enumerate(ncomp):
        mask[j, K - n:] = 1.0
    return lp * mask, mask


# ----------------------------------------------------------------------------- rollout
class APIOpponent:
    """Team B played by an API LLM (e.g. Gemma 4 31B on OpenRouter) through a server running --opponent external.
    Its decisions are never training data. Any failed call (e.g. 402 out of credit) aborts the rollout, so a
    game with broken opponent moves is never trained on."""
    def __init__(self, spec, api_key, temperature=0.7, max_tokens=1536, workers=40):
        from openai import OpenAI
        base_url, self.model = spec.rsplit(":", 1)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=0)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.pool = ThreadPoolExecutor(workers)
        self.calls = self.errors = 0
        self.system = None   # set from the external server's /info (model-vs-model system prompt)

    def _one(self, prompt):
        err = None
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": self.system}, {"role": "user", "content": prompt}])
                return r.choices[0].message.content or ""
            except Exception as e:
                err = e
                if "402" in str(e) or "401" in str(e):
                    break
                time.sleep(min(30, 2 + 3 * attempt))
        raise RuntimeError(f"opponent API failed: {type(err).__name__}: {err}")

    def act(self, prompts):
        self.calls += len(prompts)
        return list(self.pool.map(self._one, prompts))


def _new_ep(scen, team):
    return {"scenario": scen, "team": team, "states": [], "rewards": [], "own": [], "smac": [],
            "decisions": [], "info": {}}


def rollout(client, model, tok, n_episodes, temperature, max_new_tokens, scenarios=None, eval_sampling=False,
            opponent=None):
    """Plays n episodes. Against the built-in AI returns one record per game (team A). In self-play
    (server --opponent external) the same current model also plays team B, and a second record per game
    (team "B", with B's own rewards/credit/validity) is returned; both teams' decisions are training data."""
    pool = ThreadPoolExecutor(client.n)
    episodes, done_n = [], 0
    while done_n < n_episodes:
        k = min(client.n, n_episodes - done_n)
        scen = [scenarios[done_n + i] if scenarios else None for i in range(k)]
        states = list(pool.map(lambda i: client.post("/reset", {"env": i, "scenario": scen[i]}), range(k)))
        for st in states:
            if "error" in st:
                raise RuntimeError(st["error"] + st.get("trace", ""))
        selfplay = "prompts_b" in states[0]
        eps = {("A", i): _new_ep(scen[i], "A") for i in range(k)}
        if selfplay:
            eps.update({("B", i): _new_ep(scen[i], "B") for i in range(k)})
        for i in range(k):
            eps[("A", i)]["opponent"] = "api" if (selfplay and opponent is not None) else ("self" if selfplay else "builtin")
        live = set(range(k))
        while live:
            batch = [("A", i, a, SysPrompt(p, opponent.system) if opponent is not None else p)
                     for i in sorted(live) for a, p in states[i]["prompts"].items()]
            per_env = {}
            if selfplay and opponent is not None:   # team B = API model, run while the GPU generates for A
                bb = [(i, a, p) for i in sorted(live) for a, p in states[i]["prompts_b"].items()]
                fut = opponent.pool.submit(opponent.act, [p for *_, p in bb])
            elif selfplay:
                batch += [("B", i, a, p) for i in sorted(live) for a, p in states[i]["prompts_b"].items()]
            gen = generate_ids(model, tok, [p for *_, p in batch], temperature, max_new_tokens, eval_sampling)
            if selfplay and opponent is not None:
                for (i, a, p), text in zip(bb, fut.result()):
                    per_env.setdefault(("B", i), {})[a] = text
            for (tm, i, a, p), (ids, lp, text) in zip(batch, gen):
                per_env.setdefault((tm, i), {})[a] = text
                e = eps[(tm, i)]
                assert not (tm == "B" and opponent is not None)
                e["decisions"].append({"t": len(e["rewards"]), "agent": a, "prompt": p, "ids": ids, "lp": lp,
                                       "reply": text})
            for i in sorted(live):
                eps[("A", i)]["states"].append(states[i].get("state"))
            res = list(pool.map(lambda i: (i, client.post("/step", {"env": i, "replies": per_env.get(("A", i), {}),
                                                                     "replies_b": per_env.get(("B", i), {})})),
                                sorted(live)))
            for i, st in res:
                if "error" in st:
                    raise RuntimeError(st["error"] + st.get("trace", ""))
                inf = st["info"]
                teams = [("A", "reward", "reward_unit", "valid", inf.get("reward_smac", st["reward"]))]
                if selfplay:
                    teams.append(("B", "reward_b", "reward_unit_b", "valid_b", inf.get("reward_smac_b", st.get("reward_b", 0.0))))
                for tm, rk, uk, vk, sm in teams:
                    e = eps[(tm, i)]
                    t = len(e["rewards"])
                    e["rewards"].append(float(st.get(rk, 0.0)))
                    e["own"].append({str(a): float(v) for a, v in (st.get(uk) or {}).items()})
                    e["smac"].append(float(sm))
                    for d in e["decisions"]:
                        if d["t"] == t:
                            d["valid"] = bool((st.get(vk) or {}).get(d["agent"], True))
                states[i] = st
                if st["done"]:
                    eps[("A", i)]["info"] = {k_: inf.get(k_) for k_ in ("battle_won", "dead_allies", "dead_enemies",
                                                                       "episode_limit", "winner")}
                    if selfplay:
                        eps[("B", i)]["info"] = {"battle_won": inf.get("winner") == "B", "winner": inf.get("winner"),
                                                 "dead_allies": inf.get("dead_enemies"),
                                                 "dead_enemies": inf.get("dead_allies")}
                    live.discard(i)
        episodes += [eps[("A", i)] for i in range(k)] + \
            ([eps[("B", i)] for i in range(k)] if selfplay and opponent is None else [])
        done_n += k
    pool.shutdown()
    return episodes


def ep_stats(eps):
    n = max(len(eps), 1)
    dec = [d for e in eps for d in e["decisions"]]
    return {"episodes": len(eps),
            "win_rate": round(sum(bool(e["info"].get("battle_won")) for e in eps) / n, 3),
            "smac_return": round(sum(sum(e["smac"]) for e in eps) / n, 3),
            "shaped_return": round(sum(sum(e["rewards"]) for e in eps) / n, 3),
            "dead_allies": round(sum(e["info"].get("dead_allies") or 0 for e in eps) / n, 2),
            "dead_enemies": round(sum(e["info"].get("dead_enemies") or 0 for e in eps) / n, 2),
            "survival": round(1 - sum(e["info"].get("dead_allies") or 0 for e in eps) / (5 * n), 3),
            "invalid_rate": round(sum(not d.get("valid", True) for d in dec) / max(len(dec), 1), 4),
            "msg_rate": round(sum('"msg"' in (d["reply"] or "") for d in dec) / max(len(dec), 1), 3),
            "steps": round(sum(len(e["rewards"]) for e in eps) / n, 1)}


# ----------------------------------------------------------------------------- critic
class Critic(nn.Module):
    def __init__(self, sdim, n_agents, hidden=256):
        super().__init__()
        self.n = n_agents
        self.register_buffer("mu", torch.zeros(sdim))
        self.register_buffer("sd", torch.ones(sdim))
        self.net = nn.Sequential(nn.Linear(sdim + n_agents, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, s, u):
        x = torch.cat([(s - self.mu) / self.sd, nn.functional.one_hot(u, self.n).float()], -1)
        return self.net(x).squeeze(-1)


def gae(eps, critic, n_agents, gamma, lam, invalid_penalty, dev):
    """Per-agent GAE on the joint timeline. Returns rows for critic training and actor samples."""
    rows, samples = [], []
    for e in eps:
        T = len(e["rewards"])
        S = torch.tensor(e["states"], dtype=torch.float32, device=dev)          # (T, sdim)
        with torch.no_grad():
            V = torch.stack([critic(S, torch.full((T,), u, device=dev, dtype=torch.long))
                             for u in range(n_agents)], 1).cpu()                   # (T, n)
        pen = torch.zeros(T, n_agents)
        dmap = {}
        for d in e["decisions"]:
            u = int(d["agent"])
            dmap[(d["t"], u)] = d
            if not d.get("valid", True):
                pen[d["t"], u] = invalid_penalty
        for u in range(n_agents):
            adv, last = [0.0] * T, 0.0
            for t in range(T - 1, -1, -1):
                nxt = V[t + 1, u].item() if t + 1 < T else 0.0           # episode end is terminal
                r = e["rewards"][t] + e["own"][t].get(str(u), 0.0) - pen[t, u].item()
                delta = r + gamma * nxt - V[t, u].item()
                last = delta + gamma * lam * last
                adv[t] = last
            for t in range(T):
                ret = adv[t] + V[t, u].item()
                rows.append((e["states"][t], u, ret, V[t, u].item()))
                if (t, u) in dmap:
                    samples.append((dmap[(t, u)], adv[t]))
    return rows, samples


def grpo_advantages(eps, gamma, invalid_penalty, sd_floor):
    """Critic-free GRPO baseline. Group = the G rollouts of the same (scenario, team).
    A_t^u = (R_i - mean_g R) / max(std_g R, floor) + G^u_t(own credit) / max(std_g R, floor) - invalid penalty,
    R_i = episode team return (reward v2, includes the +-2 outcome), G^u_t = discounted return-to-go of the
    unit's own zero-sum credit. Whole-episode returns avoid the bias of comparing return-to-go of
    rollouts that already ended."""
    groups = {}
    for e in eps:
        groups.setdefault((e["scenario"], e["team"]), []).append(e)
    samples, kept = [], 0
    for g in groups.values():
        if len(g) < 2:
            continue
        R = [sum(e["rewards"]) for e in g]
        mu = sum(R) / len(R)
        sd = (sum((r - mu) ** 2 for r in R) / (len(R) - 1)) ** 0.5
        den = max(sd, sd_floor)
        kept += 1
        for e, r in zip(g, R):
            T = len(e["rewards"])
            own = {}
            for u in {d["agent"] for d in e["decisions"]}:
                acc, out = 0.0, [0.0] * T
                for t in range(T - 1, -1, -1):
                    acc = e["own"][t].get(u, 0.0) + gamma * acc
                    out[t] = acc
                own[u] = out
            for d in e["decisions"]:
                a = (r - mu) / den + own[d["agent"]][d["t"]] / den
                if not d.get("valid", True):
                    a -= invalid_penalty
                samples.append((d, a))
    return samples, kept, len(groups)


def train_critic(critic, opt, rows, epochs, dev, bs=512):
    S = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=dev)
    U = torch.tensor([r[1] for r in rows], dtype=torch.long, device=dev)
    R = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=dev)
    Vold = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    ev = 1 - float(torch.var(R.cpu() - Vold) / (torch.var(R.cpu()) + 1e-8))   # before the update
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(rows), device=dev)
        for i in range(0, len(rows), bs):
            idx = perm[i:i + bs]
            loss = ((critic(S[idx], U[idx]) - R[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    return ev, sum(losses[-max(1, len(losses) // epochs):]) / max(1, len(losses) // epochs)


# ----------------------------------------------------------------------------- main loop
def mappo_critic_step(args, it, eps, critic, copt, n_agents, rec, dev):
    """Normalise states (fixed after the first batch), per-agent GAE, then fit the critic. Returns actor samples."""
    # critic statistics: normaliser from this batch (first iteration), then slow EMA
    S_all = torch.tensor([s for e in eps for s in e["states"]], dtype=torch.float32, device=dev)
    assert torch.isfinite(S_all).all(), "non-finite global state"
    if it == 0 and not args.critic_init:   # fixed normaliser (set once from the first batch, never updated)
        critic.mu.copy_(S_all.mean(0)); critic.sd.copy_(S_all.std(0).clamp(min=0.05))
        json.dump({"mu": critic.mu.tolist(), "sd": critic.sd.tolist(), "min": S_all.min(0).values.tolist(),
                   "max": S_all.max(0).values.tolist()}, open(Path(args.out) / "state_norm.json", "w"))
    rec.update(state_min=round(float(S_all.min()), 3), state_max=round(float(S_all.max()), 3),
               state_shift=round(float(((S_all.mean(0) - critic.mu) / critic.sd).abs().mean()), 3))
    rows, samples = gae(eps, critic, n_agents, args.gamma, args.lam, args.invalid_penalty, dev)
    c_before = [q.detach().clone() for q in critic.net.parameters()]
    ev, vloss = train_critic(critic, copt, rows, args.critic_epochs, dev)
    rec["critic_delta"] = round(math.sqrt(sum(float(((q.detach() - q0) ** 2).sum())
                                              for q, q0 in zip(critic.net.parameters(), c_before))), 5)
    A = torch.tensor([a for _, a in samples])
    rec.update(critic_ev=round(ev, 3), value_loss=round(vloss, 4), n_samples=len(samples),
               adv_raw_mean=round(float(A.mean()), 4), adv_raw_std=round(float(A.std()), 4),
               adv_p05=round(float(A.quantile(0.05)), 3), adv_p95=round(float(A.quantile(0.95)), 3))
    return samples


def lora_params(model, name="default"):
    return {n: p for n, p in model.named_parameters() if "lora_" in n and f".{name}." in n}


def ref_checksum(model):
    return round(sum(float(p.detach().double().abs().sum()) for p in lora_params(model, "ref").values()), 6)


def evaluate(args, client, model, tok, it, log):
    """Fixed scenarios vs the BUILT-IN AI, team A only, train.py's evaluation sampling (T=0.7, top_k=64)."""
    model.eval()
    model.config.use_cache = True
    scen = [args.eval_scenarios + j for j in range(args.eval_episodes)]
    eps = rollout(client, model, tok, len(scen), 0.7, args.max_new_tokens, scen, eval_sampling=True)
    st = {"mode": "eval", "iter": it, "scenarios": [scen[0], scen[-1]], **ep_stats(eps),
          "wins": [int(bool(e["info"].get("battle_won"))) for e in eps]}
    print("eval", json.dumps(st), flush=True)
    log.write(json.dumps(st) + "\n"); log.flush()
    with open(Path(args.out) / "eval_episodes.jsonl", "a") as f:
        for e in eps:
            f.write(json.dumps({"iter": it, **{k: v for k, v in e.items() if k != "states"}}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", required=True, help="SFT LoRA to start from (also the frozen KL reference)")
    p.add_argument("--env-url", required=True)
    p.add_argument("--env-token-file", default="data/env_token")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--invalid-penalty", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl", type=float, default=0.02, help="k3 KL to the frozen SFT adapter")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-decisions", type=int, default=2048, help="actor samples per iteration (uniform)")
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--critic-warmup", type=int, default=1, help="iterations that train only the critic")
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--eval-scenarios", type=int, default=100000)
    p.add_argument("--eval-first", action="store_true")
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--save-episodes", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algo", choices=["mappo", "grpo"], default="mappo",
                   help="mappo: centralised critic + GAE; grpo: critic-free group baseline (G rollouts per scenario)")
    p.add_argument("--group-size", type=int, default=4, help="grpo: rollouts per scenario")
    p.add_argument("--adv-sd-floor", type=float, default=0.5, help="grpo: lower bound on the group std")
    p.add_argument("--opp-env-url", default=None, help="mixed opponents: a server with --opponent external whose "
                   "team B is played by --opp-llm; --opp-frac of each round's groups are played there")
    p.add_argument("--opp-llm", default="https://openrouter.ai/api/v1:google/gemma-4-31b-it")
    p.add_argument("--opp-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--opp-frac", type=float, default=0.5)
    p.add_argument("--eval-env-url", default=None, help="built-in-AI server for evaluation (required in self-play)")
    p.add_argument("--ref-adapter", default=None, help="frozen KL reference (default: --adapter). Set to the SFT "
                   "adapter when resuming from an RL checkpoint")
    p.add_argument("--critic-init", default=None, help="resume: critic_itN.pt (includes the fixed state normaliser)")
    p.add_argument("--start-iter", type=int, default=0, help="resume: iteration number of the loaded checkpoint")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(Path(args.out) / "config.json", "w"), indent=1)
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"

    tok, model = load_model(args.model, args.adapter, 16, train=True)
    model.load_adapter(args.ref_adapter or args.adapter, adapter_name="ref", is_trainable=False)
    model.set_adapter("default")
    actor_params = [p_ for p_ in lora_params(model).values()]
    for p_ in lora_params(model, "ref").values():
        p_.requires_grad_(False)
    assert actor_params and all(p_.requires_grad for p_ in actor_params)
    init = {n: p_.detach().clone() for n, p_ in lora_params(model).items()}
    init_norm = math.sqrt(sum(float((v.float() ** 2).sum()) for v in init.values()))
    opt = torch.optim.AdamW(actor_params, lr=args.lr)
    ref0 = ref_checksum(model)
    assert lora_params(model, "ref") and abs(ref0 - init_norm) >= 0, "reference adapter missing"

    client = EnvClient(args.env_url, open(args.env_token_file).read().strip())
    info = client.post("/info", {})
    SYSTEM.update(text=info.get("system", SYSTEM_PROMPT), channel=info.get("channel", "none"))
    GEN["max_new_tokens"] = args.max_new_tokens
    print("env:", json.dumps({k: v for k, v in info.items() if k != "system"}), flush=True)
    st0 = client.post("/reset", {"env": 0, "scenario": 1})
    selfplay = "prompts_b" in st0
    eval_client = EnvClient(args.eval_env_url, open(args.env_token_file).read().strip()) if args.eval_env_url else client
    assert eval_client.post("/info", {}).get("opponent") == "builtin", "evaluation must be against the built-in AI"
    opp_client = opponent = None
    if args.opp_env_url:
        assert not selfplay, "--env-url must be the built-in-AI server when --opp-env-url is used"
        opp_client = EnvClient(args.opp_env_url, open(args.env_token_file).read().strip())
        oinfo = opp_client.post("/info", {})
        assert oinfo.get("opponent") == "external", "--opp-env-url needs --opponent external"
        for k_ in ("channel", "brief", "reward", "map", "difficulty"):
            assert oinfo.get(k_) == info.get(k_), f"servers differ in {k_}"
        opponent = APIOpponent(args.opp_llm, os.environ[args.opp_key_env])
        opponent.system = oinfo["system"]
        assert "ENEMY BEHAVIOUR" not in opponent.system
        print(f"mixed opponents: {args.opp_frac:.0%} of groups vs {args.opp_llm}", flush=True)
    if args.algo == "mappo":
        assert st0.get("state"), "env server must run with --global-state"
    sdim, n_agents = len(st0.get("state") or [0]), 5
    print(f"algo {args.algo}; self-play {selfplay}", flush=True)
    critic = Critic(sdim, n_agents).to(dev)
    if args.critic_init:
        critic.load_state_dict(torch.load(args.critic_init, map_location=dev))
    copt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    log = open(Path(args.out) / "train_log.jsonl", "a")
    print(f"state dim {sdim}; actor LoRA tensors {len(actor_params)}; init norm {init_norm:.3f}", flush=True)
    if args.eval_first:
        evaluate(args, eval_client, model, tok, 0, log)

    for it in range(args.start_iter, args.start_iter + args.iters):
        t0 = time.time()
        model.eval()
        model.config.use_cache = True
        scen = None
        if args.algo == "grpo":   # G rollouts of each random scenario
            ids = [random.randrange(1 << 30) for _ in range(max(1, args.episodes_per_iter // args.group_size))]
            scen = [x for x in ids for _ in range(args.group_size)]
        if opp_client is not None:   # mixed: part of the groups vs the API model, the rest vs the built-in AI
            assert scen, "mixed opponents need --algo grpo"
            n_api = round(len(scen) // args.group_size * args.opp_frac) * args.group_size
            eps = rollout(client, model, tok, len(scen) - n_api, args.temperature, args.max_new_tokens,
                          scen[n_api:]) if len(scen) > n_api else []
            if n_api:
                eps += rollout(opp_client, model, tok, n_api, args.temperature, args.max_new_tokens, scen[:n_api],
                               opponent=opponent)
        else:
            eps = rollout(client, model, tok, len(scen) if scen else args.episodes_per_iter, args.temperature,
                          args.max_new_tokens, scen)
        eA = [e for e in eps if e["team"] == "A"]
        rec = {"mode": "train", "algo": args.algo, "selfplay": selfplay, "iter": it, **ep_stats(eA),
               "rollout_s": round(time.time() - t0, 1)}
        if opp_client is not None:
            for o in ("builtin", "api"):
                sub = [e for e in eA if e.get("opponent") == o]
                if sub:
                    rec[f"win_rate_{o}"] = round(sum(bool(e["info"].get("battle_won") or e["info"].get("winner") == "A")
                                                     for e in sub) / len(sub), 3)
                    rec[f"n_{o}"] = len(sub)
            rec["opp_calls"] = opponent.calls
        if selfplay:
            rec.update(b_win_rate=round(sum(e["info"].get("winner") == "B" for e in eA) / len(eA), 3),
                       draw_rate=round(sum(e["info"].get("winner") == "draw" for e in eA) / len(eA), 3),
                       invalid_rate_all=ep_stats(eps)["invalid_rate"])
        print("rollout", json.dumps(rec), flush=True)
        own_sum = max((abs(sum(o.values())) for e in eps for o in e["own"] if o), default=0.0)
        rec["credit_sum_max"] = round(own_sum, 9)   # per-unit credit must sum to 0 every step
        if args.algo == "grpo":
            samples, kept, ng = grpo_advantages(eps, args.gamma, args.invalid_penalty, args.adv_sd_floor)
            A = torch.tensor([a for _, a in samples]) if samples else torch.zeros(1)
            rec.update(groups_kept=kept, groups=ng, n_samples=len(samples),
                       adv_raw_mean=round(float(A.mean()), 4), adv_raw_std=round(float(A.std()), 4),
                       adv_p05=round(float(A.quantile(0.05)), 3), adv_p95=round(float(A.quantile(0.95)), 3))
        else:
            samples = mappo_critic_step(args, it, eps, critic, copt, n_agents, rec, dev)
        if args.algo == "mappo" and it - args.start_iter < args.critic_warmup and not args.critic_init:
            rec["actor_update"] = False
        else:
            if len(samples) > args.max_decisions:
                samples = random.sample(samples, args.max_decisions)
            a = torch.tensor([x for _, x in samples])
            a = ((a - a.mean()) / (a.std() + 1e-6)).tolist()
            samples = [(d, x) for (d, _), x in zip(samples, a)]
            model.train()
            model.config.use_cache = False
            # old log-probs = the per-token values recorded AT SAMPLING (behaviour policy). A no-grad
            # recomputation with the same weights is kept only as a consistency check.
            old = [torch.tensor(d["lp"], device=dev) for d, _ in samples]
            ref, gap_tok, gap_max, dref = [], [], 0.0, []
            with torch.no_grad():
                for i in range(0, len(samples), args.batch):
                    b = [d for d, _ in samples[i:i + args.batch]]
                    lp, m = token_logprobs(model, tok, [d["prompt"] for d in b], [d["ids"] for d in b])
                    for j, d in enumerate(b):
                        n = len(d["ids"])
                        diff = (lp[j, -n:] - torch.tensor(d["lp"], device=dev)).abs()
                        gap_tok.append(float(diff.mean())); gap_max = max(gap_max, float(diff.max()))
                    model.set_adapter("ref")
                    lr_, _ = token_logprobs(model, tok, [d["prompt"] for d in b], [d["ids"] for d in b])
                    model.set_adapter("default")
                    ref += [lr_[j] for j in range(len(b))]
                    dref += [float(((lp[j] - lr_[j]) * m[j]).abs().sum() / m[j].sum()) for j in range(len(b))]
            for p_ in lora_params(model, "ref").values():
                p_.requires_grad_(False)
            pgl, kls, akl, cfs, gns = [], [], [], [], []
            n_mb = 0
            for _ in range(args.ppo_epochs):
                order = list(range(len(samples)))
                random.shuffle(order)
                for i in range(0, len(order), args.batch):
                    idx = order[i:i + args.batch]
                    b = [samples[j] for j in idx]
                    lp, m = token_logprobs(model, tok, [d["prompt"] for d, _ in b], [d["ids"] for d, _ in b])
                    K = lp.shape[1]
                    o = torch.stack([nn.functional.pad(old[j], (K - old[j].shape[0], 0))[-K:] for j in idx])
                    r_ = torch.stack([nn.functional.pad(ref[j], (K - ref[j].shape[0], 0))[-K:] for j in idx])
                    adv = torch.tensor([x for _, x in b], device=dev)[:, None]
                    ratio = torch.exp((lp - o) * m)
                    surr = torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)
                    tokmean = lambda x: ((x * m).sum(-1) / m.sum(-1).clamp(min=1)).mean()
                    pg = -tokmean(surr)
                    lr_ = (r_ - lp) * m
                    kl = tokmean(torch.exp(lr_) - lr_ - 1)
                    loss = pg + args.kl * kl
                    (loss / args.grad_accum).backward()
                    n_mb += 1
                    if n_mb % args.grad_accum == 0 or i + args.batch >= len(order):
                        gns.append(float(nn.utils.clip_grad_norm_(actor_params, 1.0)))
                        opt.step()
                        opt.zero_grad()
                    pgl.append(pg.item()); kls.append(kl.item())
                    akl.append(float(tokmean((o - lp).detach())))
                    cfs.append(float((((ratio - 1).abs() > args.clip).float() * m).sum() / m.sum()))
            delta = math.sqrt(sum(float(((p_.detach() - init[n]).float() ** 2).sum())
                                  for n, p_ in lora_params(model).items()))
            rec.update(actor_update=True, n_actor=len(samples), updates=len(gns),
                       policy_loss=round(sum(pgl) / len(pgl), 4), kl_sft=round(sum(kls) / len(kls), 4),
                       approx_kl_old=round(sum(akl) / len(akl), 5), clipfrac=round(sum(cfs) / len(cfs), 4),
                       grad_norm=round(sum(gns) / max(len(gns), 1), 4),
                       lora_delta=round(delta, 5), lora_delta_rel=round(delta / max(init_norm, 1e-8), 6),
                       sampled_vs_recomputed_tok=round(sum(gap_tok) / max(len(gap_tok), 1), 5),
                       sampled_vs_recomputed_max=round(gap_max, 4),
                       default_vs_ref_tok=round(sum(dref) / max(len(dref), 1), 5),
                       ref_unchanged=ref_checksum(model) == ref0)
            del old, ref
            torch.cuda.empty_cache()
        if (it + 1) % args.save_every == 0 or it == args.start_iter + args.iters - 1:
            model.save_pretrained(Path(args.out) / f"adapter_it{it + 1}", selected_adapters=["default"])
            if args.algo == "mappo":
                torch.save(critic.state_dict(), Path(args.out) / f"critic_it{it + 1}.pt")
        rec["iter_s"] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if args.save_episodes:
            with open(Path(args.out) / "episodes.jsonl", "a") as f:
                for e in eps:
                    f.write(json.dumps({"iter": it, **e}) + "\n")
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            evaluate(args, eval_client, model, tok, it + 1, log)


if __name__ == "__main__":
    main()
