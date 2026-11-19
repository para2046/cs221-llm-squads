#!/usr/bin/env bash
# Modal part of the unified protocol: E2 (untrained 4B, 64 games, fixed scenarios, 4 cells in parallel) and
# E3 reward v2.1 -> E4 -> E4b. Results land in the smac-runs volume; fetched into runs/ at the end.
cd ~/cs221/project
M=.venv-modal/bin/modal
EV="--mode eval --temperature 0.7 --max-new-tokens 64 --save-episodes"
echo "$(date +%H:%M) E3 v2.1 start"
$M run --detach smac_llm/modal_app.py::selfplay_main --name u_v21 --gen-bs 64 \
  --server-args "--opponent external --brief unified --channel struct --reward shaped21" \
  --eval-server-args "--opponent builtin --brief unified --channel struct" \
  --args "--mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens 64 --batch 4 --grad-accum 32 --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://127.0.0.1:8391 --iters 20 --group-size 4 --episodes-per-iter 8 --save-every 2 --eval-every 4 --eval-episodes 16 --adv-sd-floor 0.5 --out runs/u_v21" > logs/modal_u_v21.out 2>&1
N=$($M volume ls smac-runs /runs/u_v21 2>/dev/null | grep -o "adapter_it[0-9]*" | sed "s/adapter_it//" | sort -n | tail -1)
A=runs/u_v21/adapter_it$N; echo "$(date +%H:%M) E3 v2.1 done, final adapter $A"
$M run --detach smac_llm/modal_app.py::main --name e4_u_v21 --server-args "--opponent builtin --brief unified --channel struct" \
   --args "$EV --adapter $A --iters 8 --episodes-per-iter 8 --eval-scenarios 100000 --out runs/e4_u_v21" > logs/modal_e4_u_v21.out 2>&1 &
for T in B A; do
  $M run --detach smac_llm/modal_app.py::main --name e4b_u_v21_base$T --server-args "--opponent external --brief unified --channel struct" \
     --args "$EV --adapter $A --iters 2 --episodes-per-iter 8 --eval-scenarios 200000 --base-team $T --out runs/e4b_u_v21_base$T" > logs/modal_e4b_u_v21_base$T.out 2>&1 &
done
wait
for d in u_v21 e4_u_v21 e4b_u_v21_baseB e4b_u_v21_baseA; do
  $M volume get --force smac-runs /runs/$d runs/ >/dev/null 2>&1; $M volume get --force smac-runs /logs/$d.log logs/ >/dev/null 2>&1; done
echo "$(date +%H:%M) all done"
