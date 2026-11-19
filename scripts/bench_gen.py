"""Time batched generation for Gemma 4 E4B under the trainer's settings."""
import time, torch, sys
sys.path.insert(0, ".")
from smac_llm.train import load_model, generate
P = open("scripts/sample_prompt.txt").read()
for train in (False, True):
    tok, model = load_model("google/gemma-4-E4B-it", None, 16, train=train)
    model.eval(); model.config.use_cache = True
    for bs in (8, 40):
        torch.cuda.synchronize(); t = time.time()
        out = generate(model, tok, [P] * bs, 1.0, max_new_tokens=24, bs=64)
        torch.cuda.synchronize()
        print(f"train={train} bs={bs} {time.time()-t:.2f}s sample={out[0]!r}", flush=True)
    del model; torch.cuda.empty_cache()
