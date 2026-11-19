#!/usr/bin/env bash
# 4B vs 4B self-play RL on the held GPU, in the best prompt/comm condition from the E2 cells.
# On-policy GRPO (both teams' decisions, shaped symmetric reward) + off-policy self-imitation of past wins;
# every 3 iterations the current weights play 16 games vs the built-in AI on a second env server.
cd ~/cs221/project
while pgrep -f "scripts/cluster_e2b_4b\.sh" >/dev/null; do sleep 60; done
read BRIEF CHAN MNT <<< $(python3 - <<'PY'
import json, glob
cells = {"e2_4b_facts_vision": ("facts", "info", 64), "e2_4b_facts_struct": ("facts", "struct", 64),
         "e2_4b_rules_info": ("rules", "info", 64), "e2_4b_rules_struct2": ("rules", "struct", 128),
         "e2_4b_facts_struct2": ("facts", "struct", 128)}
best = None
for c, cfg in cells.items():
    try:
        r = [json.loads(l) for l in open(f"runs/{c}/eval_log.jsonl")]
    except FileNotFoundError:
        continue
    n = sum(x["episodes"] for x in r)
    key = (sum(x["win_rate"] * x["episodes"] for x in r) / n, sum(x["return"] * x["episodes"] for x in r) / n)
    if best is None or key > best[0]:
        best = (key, cfg)
print(*best[1])
PY
)
echo "$(date +%H:%M) condition: brief=$BRIEF channel=$CHAN max_new_tokens=$MNT"
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8395 8 \
   --opponent external --brief $BRIEF --channel $CHAN --reward shaped > logs/cl_env_selfplay.log 2>&1 &
S1=$!
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 8 scripts/cluster_env_server.sh 8396 8 \
   --opponent builtin --brief $BRIEF --channel $CHAN > logs/cl_env_selfplay_eval.log 2>&1 &
S2=$!
until grep -q "env server ready" logs/cl_env_selfplay.log && grep -q "env server ready" logs/cl_env_selfplay_eval.log; do
  sleep 5; kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { echo "server failed"; exit 1; }; done
G="srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh"
COMMON="--mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens $MNT --batch 8 --grad-accum 16 \
        --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://wheat-05:8396"
echo "$(date +%H:%M) smoke test"
$G selfplay_smoke http://wheat-05:8395 $COMMON --out runs/selfplay_smoke --iters 1 --group-size 2 \
   --episodes-per-iter 4 --save-every 1 --eval-every 1 --eval-episodes 4 || { echo "smoke failed"; kill $S1 $S2; exit 1; }
echo "$(date +%H:%M) full run"
$G selfplay_grpo_v1 http://wheat-05:8395 $COMMON --out runs/selfplay_grpo_v1 --iters 30 --group-size 4 \
   --episodes-per-iter 16 --save-every 3 --eval-every 3 --eval-episodes 16
echo "$(date +%H:%M) done"; kill $S1 $S2
