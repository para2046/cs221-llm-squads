"""Check that seq_logprobs is independent of batch padding."""
import sys, torch
sys.path.insert(0, ".")
from smac_llm.train import load_model, seq_logprobs, chat_prefix
tok, model = load_model("google/gemma-4-E4B-it", None, 16, train=True)
model.eval()
P = open("scripts/sample_prompt.txt").read()
short = "Step 0. You are agent a0. Legal actions: stop, move_north"
c = '{"action": "stop"}'
with torch.no_grad():
    alone = seq_logprobs(model, tok, [chat_prefix(tok, short)], [c])
    padded = seq_logprobs(model, tok, [chat_prefix(tok, short), chat_prefix(tok, P)], [c, c])
print(f"alone={alone.item():.4f} in_padded_batch={padded[0].item():.4f} diff={abs(alone.item()-padded[0].item()):.4f}")
