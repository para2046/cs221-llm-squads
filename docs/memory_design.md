# Long-context experience memory (design, phase 2) / 长上下文经验记忆（设计稿，第二阶段）

Goal: let the model look back over its own past games (actions, outcomes, reward breakdowns) and learn from them **without weight updates**, then compare with, and combine with, RL weight updates.
目标：让模型回看自己过去的对局（动作、结果、奖励分解），在**不更新权重**的情况下吸取经验；再和 RL 权重更新做对比、做结合。

Related work: Reflexion (Shinn et al., 2023, arXiv 2303.11366), ExpeL (Zhao et al., 2023, arXiv 2308.10144), Voyager skill library (Wang et al., 2023, arXiv 2305.16291).

## 1. What is stored / 存什么

After every game, one **game record** (JSON, ~300-600 tokens), written by code, not the model:
每局结束后，由代码（不是模型）写一条对局记录：
- setting: unit types on both sides, start positions (coarse: "we west, they east"), brief/channel.
  设置：双方兵种、起始位置（粗略）、说明和通信方式。
- outcome: win/loss/draw, steps, units lost/killed, SMAC return, shaped-reward parts (hp, kill, death, outcome, focus).
  结果：胜/负/平、步数、阵亡和击杀、SMAC return、各项奖励分解。
- timeline of key events: first contact, each kill and death with step and unit types, who focused whom.
  关键事件时间线：首次接敌、每次击杀和阵亡（步数、兵种）、谁集火了谁。
- behaviour metrics: attack share, focus rate, moves during cooldown, focus/plan messages used.
  行为指标：攻击占比、集火率、冷却中移动、focus/plan 的使用。

Then one **lesson** (≤60 words), written by the model itself in a separate "coach" call that reads the record:
然后由模型自己在一次单独的"教练"调用里读这条记录，写一条教训（≤60 词）：
"What decided this game? What should the team do differently next time?"

## 2. How it is used / 怎么用

Two levels, so per-step inference stays fast:
分两层，保证每步推理仍然快：
- **Between games (long context) / 局与局之间（长上下文）**：a coach call reads ALL past game records and lessons (tens of thousands of tokens; Gemma 4 E4B has a 128k window) plus the aggregate statistics (win rate by unit mix, which tactics correlated with winning), and writes a **team playbook** (≤200 words).
  教练调用读取**全部**历史对局记录和教训（几万 token；Gemma 4 E4B 支持 128k 上下文）以及汇总统计（按兵种组合的胜率、哪些打法和胜利相关），写出一份**团队战术手册**（≤200 词）。
- **During a game (short) / 对局中（短）**：the playbook is appended to every unit's system prompt. No per-step retrieval, so each step costs only ~200 extra tokens.
  手册加进每个单位的系统指令。对局中不做检索，每步只多约 200 token。
- Optional retrieval / 可选检索：add the 2 most similar past games (same unit mix) as short examples.
  加入兵种组合最相似的 2 局过去对局作为简短例子。

## 3. Experiments / 实验

Same seeds, 40-80 games in sequence (memory grows as games are played):
同样的种子，按顺序打 40-80 局（记忆随对局增长）：
1. no memory (baseline) / 无记忆（基线）
2. records only (playbook from statistics, no lessons) / 只有记录
3. records + model-written lessons + playbook / 记录 + 模型写的教训 + 手册
4. memory + RL-trained weights (combined) / 记忆 + RL 训练后的权重

Measure: win rate in windows of 10 games (a learning curve without weight updates), and whether the playbook's advice shows up in behaviour metrics (e.g. focus rate rises after the playbook says "focus").
指标：每 10 局一个窗口的胜率（不更新权重的学习曲线），以及手册的建议是否体现在行为指标里（比如手册说"集火"之后集火率是否上升）。

Controls: a shuffled-memory control (records from other unit mixes) to check the gain comes from relevant experience, not just longer prompts.
对照：打乱的记忆（来自其他兵种组合的记录），检验提升来自相关经验，而不只是提示变长。

## 4. Implementation sketch / 实现要点

- `smac_llm/memory.py`: `GameRecord.from_episode(log)`, `coach_lesson(record)`, `playbook(records)`; storage `runs/<run>/memory.jsonl`.
- `Game(..., playbook=str)` appends it to the system prompt; `play.py --memory runs/<run>/memory.jsonl --coach-model <spec>`.
- Works for API models (gpt-oss, 31B) and for the 4B on the GPU (coach = same model, or a stronger API model as an ablation).
  API 模型和 GPU 上的 4B 都适用（教练用同一个模型，或者用更强的 API 模型作消融）。

Status: design only; to be built after phase 1 (D1-D3 + self-play RL).
状态：仅设计；第一阶段结束后实现。
