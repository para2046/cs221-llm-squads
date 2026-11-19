"""Run SMAC-LLM training on Modal while no cluster GPU is free.

One container holds both halves of the online loop that the cluster splits across two nodes:
the SMACv2 env server (StarCraft II on CPU, its own Python 3.10 venv at /opt/smac) and the
learner (Gemma + LoRA on the GPU), talking over localhost. Outputs land in the `smac-runs`
volume under the same relative paths as on the cluster (runs/..., logs/...).

Secret `openrouter` (OPENROUTER_API_KEY) is attached for LLM opponents / distillation teachers;
create it once with: modal secret create openrouter OPENROUTER_API_KEY=...
Volumes: smac-sc2 (sc2.tar: StarCraft II 4.10 + SMAC maps), smac-hf (HF cache with Gemma),
smac-runs (runs/ and logs/).

  # one-time: download Gemma into the volume
  .venv-modal/bin/modal run smac_llm/modal_app.py::prepare
  # copy a local adapter up, to continue from it
  .venv-modal/bin/modal volume put smac-runs runs/rl_selfplay_v2_from_it20/adapter_it6 /runs/rl_selfplay_v2_from_it20/adapter_it6
  # train (detached: keeps running if this terminal closes)
  .venv-modal/bin/modal run --detach smac_llm/modal_app.py::main --name rl_modal_v3 \
      --args "--mode online --adapter runs/... --out runs/rl_modal_v3 --iters 10 ..."
  # model vs model: E2B learner against Gemma 31B over OpenRouter, team chat on
  .venv-modal/bin/modal run --detach smac_llm/modal_app.py::main --name e2b_vs_31b \
      --server-args "--opponent llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it --opponent-key-env OPENROUTER_API_KEY --channel team" \
      --args "--mode online --model google/gemma-4-E2B-it --algo grpo --out runs/e2b_vs_31b ..."
  # fetch results
  .venv-modal/bin/modal volume get smac-runs /runs/rl_modal_v3 runs/ ; ... /logs/rl_modal_v3.log logs/
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# first available of these (L40S matches the cluster; Modal often runs short of L40S capacity)
GPU = ["L40S", "A100-80GB"]  # H100 is not available on the starter plan; A100-40GB OOMs in BC

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "transformers==5.23.0", "peft==0.21.3", "accelerate==1.15.0",
                 "safetensors", "numpy", "requests", "huggingface_hub", "uv")
    .add_local_dir(ROOT / "external/smacv2", "/opt/smacv2_src", copy=True,
                   ignore=[".git", "**/__pycache__", "*.egg-info"])
    .run_commands(
        "uv venv /opt/smac --python 3.10",
        "uv pip install --python /opt/smac/bin/python pysc2==4.0.0 protobuf==3.20.3 "
        "s2clientprotocol==5.0.17.98310.0 absl-py numpy==2.2.6 requests pygame openai /opt/smacv2_src",
    )
    .env({"HF_HOME": "/hf", "SC2PATH": "/sc2local/StarCraftII", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "smac_llm", "/root/project/smac_llm", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/project/scripts", ignore=["__pycache__"])
)

app = modal.App("smac-llm", image=image)
sc2_vol = modal.Volume.from_name("smac-sc2")
hf_vol = modal.Volume.from_name("smac-hf")
runs_vol = modal.Volume.from_name("smac-runs")
VOLS = {"/sc2": sc2_vol, "/hf": hf_vol, "/runs": runs_vol}


@app.function(volumes={"/hf": hf_vol}, cpu=4, memory=16384, timeout=3600)
def prepare(model: str = "google/gemma-4-E4B-it"):
    from huggingface_hub import snapshot_download
    path = snapshot_download(model)
    hf_vol.commit()
    print("downloaded", model, "to", path)


def _setup_workdir():
    """Unpack SC2 to local disk and link runs/ and logs/ to the volume."""
    if not Path("/sc2local/StarCraftII").exists():
        t0 = time.time()
        Path("/sc2local").mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", "/sc2/sc2.tar", "-C", "/sc2local"], check=True)
        print(f"unpacked SC2 in {time.time() - t0:.0f}s", flush=True)
    os.chdir("/root/project")
    for d in ("runs", "logs"):
        Path(f"/runs/{d}").mkdir(parents=True, exist_ok=True)
        if not Path(d).exists():
            os.symlink(f"/runs/{d}", d)
    Path("data").mkdir(exist_ok=True)
    Path("data/env_token").write_text(os.urandom(16).hex())


def _start_env_server(n_envs: int, server_args: str = "", port: int = 8390, logname: str = "/tmp/env_server.log"):
    log = open(logname, "w")
    proc = subprocess.Popen(
        ["/opt/smac/bin/python", "-m", "smac_llm.env_server", "--n-envs", str(n_envs),
         "--port", str(port), "--token-file", "data/env_token", *shlex.split(server_args)],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if "env server ready" in Path(logname).read_text(errors="replace"):
            print("env server ready", flush=True)
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    print(Path(logname).read_text(errors="replace")[-4000:])
    raise RuntimeError("env server failed to start")


def _commit_loop(stop: threading.Event, every: int = 120):
    while not stop.wait(every):
        try:
            runs_vol.commit()
        except Exception as e:  # keep training even if one commit fails
            print("volume commit failed:", e, flush=True)


# must not depend on local-only state: Modal re-imports this file in the container and the
# function definitions have to match
SECRETS = [modal.Secret.from_name("openrouter")]


@app.function(gpu=GPU, cpu=16, memory=65536, timeout=24 * 3600, volumes=VOLS, secrets=SECRETS)
def train(args: str, name: str, n_envs: int = 8, server_args: str = ""):
    """Run `python -m smac_llm.train <args>` against a local env server; log to logs/<name>.log.
    server_args go to smac_llm.env_server (--opponent, --channel, --incentive-a/-b, ...)."""
    _setup_workdir()
    server = _start_env_server(n_envs, server_args)
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    cmd = [sys.executable, "-m", "smac_llm.train", *shlex.split(args),
           "--env-url", "http://127.0.0.1:8390", "--env-token-file", "data/env_token"]
    print("running:", " ".join(cmd), flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(f"logs/{name}.log", "a") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("{", "rollout", "Traceback")) or "Error" in line:
                print(line, end="", flush=True)
        rc = p.wait()
    server.terminate()
    Path("logs/env_server").mkdir(exist_ok=True)
    Path(f"logs/env_server/{name}.log").write_text(Path("/tmp/env_server.log").read_text(errors="replace")[-200000:])
    stop.set()
    runs_vol.commit()
    print("exit code", rc)
    return rc


@app.local_entrypoint()
def main(args: str, name: str, n_envs: int = 8, server_args: str = ""):
    print("exit code", train.remote(args, name, n_envs, server_args))


@app.function(gpu=["A100-80GB", "L40S"], cpu=32, memory=98304, timeout=24 * 3600, volumes=VOLS)
def selfplay(args: str, name: str, server_args: str, eval_server_args: str, n_envs: int = 8, gen_bs: int = 64):
    """Self-play training: env server for the external-opponent games on :8390 and a built-in-AI eval server on
    :8391 (pass --eval-env-url http://127.0.0.1:8391 in args). No secrets needed."""
    _setup_workdir()
    s1 = _start_env_server(n_envs, server_args, 8390, "/tmp/env_server.log")
    s2 = _start_env_server(n_envs, eval_server_args, 8391, "/tmp/env_server_eval.log")
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    cmd = [sys.executable, "-m", "smac_llm.train", *shlex.split(args),
           "--env-url", "http://127.0.0.1:8390", "--env-token-file", "data/env_token"]
    print("running:", " ".join(cmd), flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", GEN_BS=str(gen_bs))
    with open(f"logs/{name}.log", "a") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("{", "rollout", "eval_fixed", "Traceback", "generate:")) or "Error" in line:
                print(line, end="", flush=True)
        rc = p.wait()
    s1.terminate(); s2.terminate()
    stop.set()
    runs_vol.commit()
    print("exit code", rc)
    return rc


@app.local_entrypoint()
def selfplay_main(args: str, name: str, server_args: str, eval_server_args: str, gen_bs: int = 64):
    print("exit code", selfplay.remote(args, name, server_args, eval_server_args, gen_bs=gen_bs))


@app.function(gpu=GPU, cpu=16, memory=65536, timeout=6 * 3600, volumes=VOLS)
def mappo(args: str, name: str, server_args: str, n_envs: int = 8, gen_bs: int = 32):
    """Run `python -m smac_llm.mappo <args>` against one local env server (e.g. a built-in-AI server for a
    pure evaluation: --iters 0 --eval-first --eval-episodes 64). Logs to logs/<name>.log on the volume."""
    _setup_workdir()
    server = _start_env_server(n_envs, server_args)
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    cmd = [sys.executable, "-m", "smac_llm.mappo", *shlex.split(args),
           "--env-url", "http://127.0.0.1:8390", "--env-token-file", "data/env_token"]
    print("running:", " ".join(cmd), flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", GEN_BS=str(gen_bs))
    with open(f"logs/{name}.log", "a") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("{", "eval", "env:", "state dim", "Traceback")) or "Error" in line:
                print(line, end="", flush=True)
        rc = p.wait()
    server.terminate()
    stop.set()
    runs_vol.commit()
    print("exit code", rc)
    return rc


@app.local_entrypoint()
def mappo_main(args: str, name: str, server_args: str, gen_bs: int = 32):
    print("exit code", mappo.remote(args, name, server_args, gen_bs=gen_bs))
