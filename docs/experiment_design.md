# Experiment design (draft for discussion) / 实验设计（讨论稿）

Status: DRAFT, nothing below is running. Decisions marked **[D#]** need the user's call.
状态：讨论稿，以下实验都还没有运行。标 **[D#]** 的地方需要你来决定。

Last updated 2026-11-16.

## 0. Fixed settings for every run / 所有实验共用的固定设置

- Map / 地图：SMACv2 `10gen_protoss`, 5 v 5, built-in AI difficulty 7. Unit types random (45% zealot, 45% stalker, 10% colossus).
  SMACv2 `10gen_protoss`，5 对 5，内置 AI 难度 7，兵种随机（狂热者 45%、追猎者 45%、巨像 10%）。
- Each unit = one LLM call per step; it sees only its own sight range (no global vision).
  每个单位每步单独调用一次模型，只能看到自己视野内的东西（没有全局视野）。
- 40 games per condition, same 40 seeds in every condition (seeds of `scripts/collect.sh`, 10 workers x 4 episodes).
  每组 40 局，所有组使用同样的 40 个种子。
- Temperature 0.7 (reasoning models that require it: 1.0).
  温度 0.7（必须用 1.0 的推理模型除外）。
- Metrics / 指标：
  - Win rate, SMAC return, units lost, illegal-order rate.
    胜率、SMAC return、我方阵亡数、非法指令率。
  - Behaviour: focus rate (steps where all attacking units hit the same target), attack share, moves during cooldown (`scripts/behavior_stats.py`).
    行为：集火率、攻击占比、冷却中移动比例。
  - With chat: message rate, plus message ablations (drop / shuffle senders) at evaluation.
    有通信时：发消息比例，以及评估时的消息消融（删掉 / 打乱发送者）。

## 1. Baselines already measured (40 games each, same seeds) / 已有基线

| Player | Win | Return | Lost | Focus |
|---|---|---|---|---|
| Focus rule bot, own vision only | 50% | 16.2 | 3.6 | 46% |
| Kiting rule bot, own vision only | 52.5% | 16.6 | 3.55 | 48% |
| Gemma 4 31B, no brief | 37.5% | 15.2 | 4.2 | 48% |
| Gemma 4 31B, facts + tactics brief | 50% | 16.5 | 3.45 | 62% |
| GPT-5.6 Luna (low reasoning), facts + tactics, 20 games | 5% | 11.1 | 4.9 | 61% (attacks only 19%) |
| Gemma 4 E4B (4B), no brief, team chat off / on | 15% / 25% | 11.8 / 13.0 | — | — |

## 2. Factors / 实验因素

**A. Prompt brief / 策略说明**
- `none`：only the 5-line system prompt. 只有 5 行系统指令。
- `facts`：rules (damage, shields, armor, cooldown), unit stats, enemy behaviour; NO tactics. 只给规则、兵种数据、敌人行为，不给战术。（待写 / to write）
- `facts+tactics`：current `STRATEGY_PROMPT`. 现在的版本。
- **[D1]** Keep all three? Add a per-unit role card ("you are a zealot: ...")? 三种都保留吗？要不要加每个兵种专属的角色卡？

**B. Communication / 通信**
- `none`：no messages. 无通信。
- `chat`：free text "say" (<=12 words), teammates read it next step (exists). 自由文本，队友下一步读到（已实现）。
- `structured`：fixed format, e.g. `FOCUS e3 | HP 40% | SEEN e1@(12,20) 60hp`, read next step. Covers "watch each other's HP" and "share vision". 固定格式，对应"互相照顾血量"和"共享视野"。（待写 / to write, ~1 h）
- `sequential`：units decide in a fixed order within the same step and see earlier units' chosen targets. 同一步内按顺序决策，后面的单位看到前面单位的目标。（待写 / to write, ~1–2 h）
- Control `shared-vision`：teammates' sightings added to each observation automatically, no talking. 对照组：自动共享视野，不说话。（待写 / to write, ~1 h）
- **[D2]** Which of these go into the report? 哪些放进报告？

**C. Player / 模型**
- Gemma 4 31B via OpenRouter (~$0.02 per game). 通过 OpenRouter。
- Gemma 4 E4B (4B) on the cluster GPU, before and after training. 4B，在集群 GPU 上，训练前和训练后。
- GPT-5.6 Luna: evaluation / demo only, never used as training data (OpenAI terms). 只评估和演示，不做训练数据。
- **[D3]** Include Luna as an evaluation row? (~$0.07 per game) 要把 Luna 作为评估的一行吗？

## 3. Proposed experiments / 拟定实验

**E1 — Gemma 31B vs built-in AI (OpenRouter)**
- Step 1: brief ablation, comm `none`: `none` (done), `facts` (new), `facts+tactics` (done). New cost ~$0.9.
  第 1 步：说明消融，无通信。新增约 $0.9。
- Step 2: with the best brief, comm `none` vs `chat` (+ `structured` / `sequential` if chosen in D2). ~$1.2 per condition.
  第 2 步：用最好的说明，比较各通信方式，每组约 $1.2。
- Step 3: message ablation on the best comm condition (drop, shuffle). ~$1.2 each.
  第 3 步：在最好的通信设置上做消息消融，每种约 $1.2。

**E2 — Gemma 4B (GPU), same cells as E1**
- Same brief ablation and comm comparison as E1, for the untrained 4B. ~25 min per condition, no API cost.
  和 E1 相同的格子，用未训练的 4B，每组约 25 分钟，不花 API 费用。

**E3 — train 4B on the 31B's winning games**
- Data: 31B wins from the best E1 condition (needs ~100+ games, ~$2–3, to have ~50 wins).
  数据：E1 最好设置下 31B 的胜局（约需 100+ 局、$2–3，得到约 50 局胜局）。
- Behaviour cloning on wins (filter), batch 4 on the L40S; evaluate in the same cell.
  只学胜局的行为克隆，L40S 上 batch 4；在同一格子里评估。
- **[D4]** With or without chat in the teacher data? 老师数据要不要带通信？

**E4 — 4B self-play**
- Both teams controlled by the 4B (current vs frozen earlier copy), GRPO, starting from the E3 model. Needs code (~half day).
  双方都由 4B 控制（当前版本对冻结的旧版本），GRPO，从 E3 的模型开始。需要写代码（约半天）。
- Reward: old SMAC vs shaped (`--reward shaped`). **[D5]** Run the reward A/B here or in E3's GRPO?
  奖励：SMAC 原版 vs 新奖励。A/B 放在这里还是 E3 之后的 GRPO？

## 4. Budget and time / 预算和时间

- OpenRouter left: ~$10.4. 剩余约 $10.4。
- GPU: one L40S held until ~Nov 18 17:00, successor requests queued. 一块 L40S，到约 11 月 18 日 17:00，后续申请在排队。
- Report due Nov 20. 报告 11 月 20 日截止。

## 5. Decisions (user, 2026-11-16 evening) / 决定

- **D1**：use `facts`: rules + unit data + enemy behaviour; tactics only as optional examples ("you decide your own tactics, e.g. ..."). Implemented as `--brief facts` (`FACTS_PROMPT`).
  用"只给事实"版本，战术只作为可选的例子，由模型自己决定。
- **D2**：structured JSON channel `--channel struct`. Each unit's status (position, hp, shield, cooldown, enemies it sees with position/hp/shield) is broadcast automatically every step, so the model spends no tokens on it; the model may add `focus` (target) and a short `plan` (<=8 words) for strategy. Controls: `none` (no talking) and `vision` (the automatic reports only = shared vision, no focus/plan).
  结构化 JSON 通道：状态自动广播（不占模型 token），模型只加 focus 和不超过 8 个词的 plan。对照组：不说话；只共享视野。
- Vision: SMACv2 default is local sight only (`fully_observable=False`; SMACv2 marks full observability "for debugging only, not a fair observation"). All our runs so far use local sight with no sharing. Team-shared vision is not a SMACv2 option; our `vision` channel adds it.
  视野：SMACv2 默认只看自己视野；之前所有实验都没有共享。团队共享视野不是 SMACv2 的选项，由我们的 vision 通道实现。
- **D3**：pilot with CS221-API models (openai/gpt-oss-120b, thinkingmachines/Inkling-2-Small, both reasoning effort low): vs built-in AI and self-play; analyse whether they form their own strategy or follow the examples.
  用课程 API 的模型做试点：对内置 AI 和自我对弈，分析是自己定策略还是照例子做。
- **D4**：decide after D1–D3. D1–D3 结束后再定。
- **D5**：no old-vs-new reward A/B; compare trained vs untrained 4B, in the same D1/D2 cells.
  不做新旧奖励对比；比较训练前后的 4B，用同样的 D1/D2 格子。
- API limit: the CS221 proxy allows 120 requests/minute per key; runs use `LLM_RPM` (10 workers x 11).
  API 限制每分钟 120 次请求；用 LLM_RPM 限速。
