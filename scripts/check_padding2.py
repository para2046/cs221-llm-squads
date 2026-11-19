import sys, torch
sys.path.insert(0, ".")
from smac_llm.train import load_model, chat_prefix
tok, model = load_model("google/gemma-4-E4B-it", None, 16, train=True); model.eval()
P = open("scripts/sample_prompt.txt").read()
short = chat_prefix(tok, "Step 0. You are agent a0. Legal actions: stop, move_north")
from smac_llm.train import end_of_turn; c = "{\"action\": \"stop\"}" + end_of_turn(tok)
def lp(prefixes, use_pos):
    ids = [tok(p + c, add_special_tokens=False)["input_ids"] for p in prefixes]
    n_c = len(tok(c, add_special_tokens=False)["input_ids"])
    L = max(map(len, ids))
    inp = torch.tensor([[tok.pad_token_id]*(L-len(x)) + x for x in ids], device="cuda")
    att = torch.tensor([[0]*(L-len(x)) + [1]*len(x) for x in ids], device="cuda")
    kw = {"position_ids": (att.cumsum(-1)-1).clamp(min=0)} if use_pos else {}
    with torch.no_grad():
        lg = model(input_ids=inp, attention_mask=att, **kw).logits[:, -n_c-1:-1].float()
    t = inp[:, -n_c:]
    return torch.log_softmax(lg, -1).gather(-1, t.unsqueeze(-1)).squeeze(-1).sum(-1)
for use_pos in (False, True):
    a = lp([short], use_pos)[0].item(); b = lp([short, chat_prefix(tok, P)], use_pos)[0].item()
    print(f"position_ids={use_pos}: alone={a:.3f} padded={b:.3f} diff={abs(a-b):.3f}")
