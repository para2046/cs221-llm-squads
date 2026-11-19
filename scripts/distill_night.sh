#!/usr/bin/env bash
# Night of 2026-11-18: 4B learns from Gemma 31B (unified protocol, C2, vs built-in AI).
#   collect  : 168 more 31B C2 games on CPU job 1992888 (12 workers x 14, seeds 20007.., disjoint from E1/E4)
#   dagger   : GPU2 (job 1991191) pure DAgger: 4B plays, 31B labels 512 visited states / iter, aggregated pool
#   awr      : GPU1 (job 1990335) offline AWR (reward v2 weights) on all 31B C2 games, then 64-game eval
cd ~/cs221/project
OR=https://openrouter.ai/api/v1:google/gemma-4-31b-it
srv() {  # jobid port cpus log
  srun --jobid $1 --overlap --ntasks 1 --cpus-per-task $3 scripts/cluster_env_server.sh $2 8 --opponent builtin \
    --brief unified --channel struct > $4 2>&1 &
  S=$!; until grep -q "env server ready" $4; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed $4"; exit 1; }; done
  H=$(grep -o "host=[^ ]*" $4 | cut -d= -f2)
}
case $1 in
collect)
  echo "$(date +%H:%M) collect start"
  SEED_BASE=20 CPU_JOB=1992888 scripts/collect_play.sh d31b_c2 12 14 --a "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" \
    --opponent builtin --brief unified --channel struct --api-key-env OPENROUTER_API_KEY --temperature 0.7
  echo "$(date +%H:%M) collect done";;
dagger)
  srv 1991191 8411 8 logs/cl_env_dagger.log
  echo "$(date +%H:%M) DAgger env on $H"
  srun --jobid 1991191 --overlap --ntasks 1 env GEN_BS=32 SKIP_EVAL0=1 scripts/cluster_train.sh dagger_c2 http://$H:8411 \
    --mode online --algo grpo --pg-coef 0 --sil-coef 0 --distill-model $OR --distill-coef 1.0 --distill-n 512 \
    --distill-aggregate --temperature 0.7 --max-new-tokens 64 --batch 4 --grad-accum 8 --ppo-epochs 1 --lr 3e-5 \
    --max-decisions 512 --save-episodes --iters 8 --group-size 4 --episodes-per-iter 8 --save-every 2 \
    --eval-env-url http://$H:8411 --eval-every 4 --eval-episodes 16 --out runs/dagger_c2
  echo "$(date +%H:%M) DAgger done"; kill $S;;
awr)
  until grep -q "collect done" logs/d31b_collect.out; do sleep 30; done
  srv 1992888 8413 8 logs/cl_env_awr.log
  echo "$(date +%H:%M) AWR env on $H"
  srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh awr_c2 http://$H:8413 --mode bc --weighting awr \
    --teacher-reward v2 --data runs/e1u_31b_struct runs/d31b_c2 --epochs 1 --batch 1 --grad-accum 8 --save-steps 4000 --min-weight 0.3 --lr 1e-4 --out runs/awr_c2
  echo "$(date +%H:%M) AWR trained"
  srun --jobid 1990335 --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh e4_awr_c2 http://$H:8413 --mode eval \
    --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter runs/awr_c2/adapter --iters 8 --episodes-per-iter 8 \
    --eval-scenarios 100000 --out runs/e4_awr_c2
  echo "$(date +%H:%M) AWR eval done"; kill $S;;
esac
