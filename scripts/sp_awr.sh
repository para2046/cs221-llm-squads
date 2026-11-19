#!/usr/bin/env bash
# 2026-11-18: self-play GRPO starting from the AWR adapter (53% vs built-in). C2 unified, reward v2 (shaped2),
# KL=0 (code's KL reference is the base 4B, which would pull back toward it), lr 1e-5, 16 games/iter, 2048 decisions,
# 64 updates/iter. GPU2 (job 1991191, until Nov 19 06:00); env servers on CPU job 1992888.
# After training: 64 games vs built-in (scenarios 100000, paired with untrained/AWR) + h2h vs untrained (8 A + 8 B).
cd ~/cs221/project
G=1991191; C=1992888
srv() {  # port cpus log opponent [extra]
  srun --jobid $C --overlap --ntasks 1 --cpus-per-task $2 scripts/cluster_env_server.sh $1 8 --opponent $4 \
    --brief unified --channel struct $5 > $3 2>&1 &
  S=$!; until grep -q "env server ready" $3; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed $3"; exit 1; }; done
  H=$(grep -o "host=[^ ]*" $3 | cut -d= -f2)
}
srv 8421 12 logs/cl_env_sp_awr.log external "--reward shaped2"; S1=$S; H1=$H
srv 8422 8 logs/cl_env_sp_awr_eval.log builtin; S2=$S; H2=$H
echo "$(date +%H:%M) sp_awr start (train env $H1, eval env $H2)"
srun --jobid $G --overlap --ntasks 1 env GEN_BS=32 SKIP_EVAL0=1 scripts/cluster_train.sh sp_awr http://$H1:8421 \
  --mode online --algo grpo --credit rtg --adapter runs/awr_c2/adapter --kl 0 --temperature 1.0 --max-new-tokens 64 \
  --batch 4 --grad-accum 8 --ppo-epochs 1 --lr 1e-5 --max-decisions 2048 --save-episodes --sil-coef 0.1 \
  --adv-sd-floor 0.5 --iters 16 --group-size 4 --episodes-per-iter 16 --save-every 2 \
  --eval-env-url http://$H2:8422 --eval-every 4 --eval-episodes 16 --out runs/sp_awr
echo "$(date +%H:%M) sp_awr trained"
A=runs/sp_awr/adapter_it$(ls -d runs/sp_awr/adapter_it* | sed "s/.*adapter_it//" | sort -n | tail -1)
echo "$(date +%H:%M) eval $A"
srun --jobid $G --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh e4_sp_awr http://$H2:8422 --mode eval \
  --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A --iters 8 --episodes-per-iter 8 \
  --eval-scenarios 100000 --out runs/e4_sp_awr
echo "$(date +%H:%M) eval done"
kill $S2
srv 8423 12 logs/cl_env_sp_awr_h2h.log external; S3=$S
for T in B A; do
  srun --jobid $G --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh h2h_sp_awr_base$T http://$H:8423 --mode eval \
    --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A --iters 1 --episodes-per-iter 8 \
    --eval-scenarios 200000 --base-team $T --out runs/h2h_sp_awr_base$T
done
echo "$(date +%H:%M) sp_awr all done"; kill $S1 $S3
