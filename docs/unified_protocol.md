# Unified protocol (draft for approval, 2026-11-17) / 统一实验协议（待确认）

Principle / 原则：every experiment uses the SAME system prompt and the SAME observation format; each experiment changes exactly one factor. Shaped rewards are used only by the training algorithm and never appear in any prompt. Our own win/loss analysis never appears in any prompt.
所有实验使用同一个系统提示和同一种观察格式，每个实验只改变一个因素。训练奖励只在算法里用，不出现在提示里；我们自己的胜负分析也不出现在提示里。

## 1. System prompt (identical for every unit, every condition, built-in AI and self-play) / 系统提示（所有条件完全相同）

```
You control ONE unit in a StarCraft II team battle. Your team shares one goal: destroy all enemy units while losing as few of your own as possible. Every teammate is controlled by a separate copy of you that sees only what its own unit sees. North is +y, east is +x.
Reply with ONLY a JSON object such as {"action": "attack_e1"}. The action must be one of the names listed under 'Legal actions'.
GAME RULES
- One step = 0.5 game seconds. A game ends when one team has no units left, or after 200 steps (100 seconds); if both teams still have units then, it is a draw.
- Your unit only shoots when you choose attack_eK; attack_eK is listed only while that enemy is within your shoot range. While you move or stop you do not shoot, even if you are being hit.
- Damage hits shields first; shields have no armor. Once shields are 0, the rest hits health, reduced by 1 armor per hit. Health never regenerates; shields regenerate 2 per second after 10 seconds without taking damage.
- After firing, your weapon cools down ("weapon cooldown", in game loops; one step = 8 loops). At 0 you can fire right away.
UNITS (health/shield; damage per attack; time between attacks; attack range; speed)
- zealot: 100/50; melee, 8 x2; 1.2 s; range 2 (must walk up to the target); speed 2.25.
- stalker: 80/80; 13, or 18 vs stalker and colossus; 1.87 s; range 6; speed 2.95.
- colossus: 200/150; 10 x2, or 15 x2 vs zealot, and the beam also hits every enemy along a short line; 1.5 s; range 7; speed 2.25.
ENEMY
- The enemy team has the same unit types as your team and follows the same rules.
TACTICS
- You decide your own tactics.
YOUR HISTORY
- Every step you also see your own last 5 steps: the action you took, what happened to your target and to your own hp+shield, and your team's game score so far (the game's default score, 0 to 20).
```
+ one channel block (the only part that differs, see §3).

Changes vs the old `facts` prompt / 和旧 facts 提示相比的改动：
- removed the tactic examples (YOUR TACTICS list) / 删除战术例子
- removed evaluative words ("your most valuable unit", "fastest unit") / 删除评价性的话
- enemy description identical for built-in AI and self-play (true in both) / 敌人描述两种对手相同
- added: time limit and draw rule; shields have no armor; attack listed only in range / 新增：时间限制和平局规则、护盾无护甲、只有在射程内才列出攻击
- added: YOUR HISTORY (per-unit memory, §2) / 新增：个人历史

## 2. Per-step observation (identical format) / 每步观察（格式相同）

```
Step 14. You are agent a0, a stalker at (14.8, 16.0), hp 62/80, shield 0/80, weapon cooldown 16, shoot range 6, sight range 10.
Visible teammates:
  a3 zealot at (16.8, 15.2), dist 2.1, hp 100/100, shield 31/50
Visible enemies:
  e3 zealot at (13.1, 17.0), dist 2.0, hp 40/100, shield 0/50 [in range]
Units alive: your team 5, enemy 4.
[Team reports: ... only in channels C1/C2]
Your last steps (oldest first):
  step 10: attack_e3 -> e3 hp+shield 120 -> 104; you 160 -> 160
  step 11: attack_e3 -> e3 104 -> 88; you 160 -> 150
  step 12: move_west -> you 150 -> 150
  step 13: attack_e3 -> e3 88 -> 40; you 150 -> 142
Team score so far: 6.3 / 20
Legal actions: stop, move_north, move_south, move_east, move_west, attack_e3
```
History: written by the environment, only game facts (own action, target hp+shield before -> after or "killed", own hp+shield before -> after), last 5 steps; team score = SMAC default reward accumulated (never the shaped training reward). Each of the 5 (or 10 in self-play) agents has its own history.
历史由环境写，只记录游戏事实（自己的动作、目标血盾变化或被击杀、自己血盾变化），最近 5 步；团队分数是 SMAC 默认分数累计（从不显示训练奖励）。每个单位各有自己的历史。

## 3. The only varied factor in prompt-condition experiments: channel / 提示类实验唯一的变量：通信

- **C0 none**：no block. 无通信。
- **C1 info**：
```
TEAM REPORTS
Every step you receive one JSON object per living teammate: its type, position, hp, shield, weapon cooldown and the enemies it can see (id, type, position, hp, shield). Your own report is sent automatically.
```
- **C2 msg**：C1 + one free-text field; the model decides what (if anything) to say. No "focus" field (it would suggest focus fire).
```
TEAM MESSAGES
You may add a "msg" field (at most 8 words) to your reply; your teammates see it in your next report. Format: {"action": "<one legal action>", "msg": "<at most 8 words>"}
```
（只有一个自由文本字段，模型自己决定说什么；不再有 focus 字段，也没有任何策略例子）

## 4. Experiments / 实验

Fixed for all: map 10gen_protoss 5v5; same seed list (64 seeds; API cells use the first 40); temperature 0.7 at evaluation; max reply tokens 64 (4B) / reasoning low (API).
所有实验固定：同一地图、同一组种子（64 个，API 用前 40 个）、评估温度 0.7。

| ID | Question / 问题 | Player | Opponent | Varied / 变量 | Games | Where |
|---|---|---|---|---|---|---|
| E1 | Does communication help? 通信有没有用 | Inkling-2-Small | built-in AI | C0 / C1 / C2 | 40 each | CS221 API (CPU) |
| E2 | Same for the small model 小模型 | Gemma 4 E4B (untrained) | built-in AI | C0 / C1 / C2 | 64 each | GPU / Modal |
| E3 | Which training reward works? 哪种奖励有效 | E4B, self-play GRPO + SIL, C2 | itself | reward v1 / v2 / v2.1 | 20 iters x 8 games | GPU1 / GPU2 / Modal |
| E4 | Did RL help? RL 有没有提升 | untrained, v1, v2, v2.1 (final) | built-in AI | weights | 64 each | GPU1 + GPU2 |

Metrics (same for all) / 指标：win rate (paired by seed, Fisher/McNemar), SMAC return, units lost; behaviour: attack share, focus rate, weakest-target share, moves during cooldown, target switching, engagement sync, regrouping; channel use (focus/plan fill rate, follow rate). Analysis only, never fed back into prompts.

Optional / 可选：Gemma 31B (OpenRouter, ~$2) C0 vs C2, 40 games each, as a second large model for E1.


## 5. History storage / 历史怎么存

- In the environment (`Game.hist[team][unit]`, last 5 lines per unit, reset each game), written after every step from the game state; inserted into that unit's next prompt before "Legal actions". Team score = accumulated SMAC default reward of that team.
- Logged: every prompt (with its history block) and reply is saved per step in the game logs (`runs/<exp>/...jsonl`), so the analysis can see exactly what each agent saw.
- The system prompt tells every agent that it sees its own last 5 steps (YOUR HISTORY).

## 6. Analysis plan (fixed before running) / 分析计划（运行前确定）

All comparisons are paired by seed (same 40/64 start scenarios). Primary test: McNemar on win/loss; secondary: paired difference in SMAC return (bootstrap 95% CI). With 40-64 games only differences of ~20+ points in win rate are detectable; smaller ones are reported as "no detectable difference".

**E1 / E2 – communication (Inkling 40 games; 4B 64 games; C0 / C1 / C2)**
- H1: shared reports help (C1 > C0). H2: free messages add something beyond reports (C2 > C1).
- Outcome: win rate, SMAC return, units lost.
- Behaviour (same script for every cell): attack share, focus rate (all attackers on one target), weakest-target share, moves during cooldown, target switching, engagement sync at first contact, regrouping after scattered spawns, per unit type (stalker kiting vs zealots, zealot engagement).
- Messages (C2): share of decisions with a msg; content categories by keywords (target call "eK", position/movement, own status/hp, plan/strategy, other); follow rate = share of teammates in range of a named enemy that attack it next step; do messages differ between won and lost games.
- "Does the model form its own strategy?": since the prompt has no tactics, any focus fire / kiting / regrouping / message plan is the model's own; compare these rates with the rule bots and between models.
- History use (descriptive): how often a unit switches action after its history shows "target unchanged" or "you lost hp".

**E3 – training (4B self-play, C2, reward v1 / v2 / v2.1, 20 iterations x 8 games)**
- Curves per iteration: SMAC return and win rate of team A in self-play (not used for conclusions), KL to base, every 4 iterations 16 games vs built-in AI.

**E4 – did RL help? (untrained, v1, v2, v2.1; 64 games vs built-in AI, C2)**
- H3: trained > untrained. H4: v2 > v1. H5: v2.1 increases engagement sync / regrouping vs v2 (and does that raise win rate?).
- Same outcome, behaviour and message metrics as E1/E2, so the report shows WHAT changed after training, not only whether win rate moved.

**Report tables / figures**: T1 communication grid (model x C0/C1/C2: win, return, focus, sync, msg use); F1 behaviour profile bars; T2 RL arms (win, return, behaviour deltas vs untrained, CIs); F2 training curves; examples of messages and of a unit's history-driven decision.

Optional extra cells (cheap): 4B C2 with history turned off (tests whether the 5-step history matters; ~$2 on Modal); Gemma 31B C0/C2 (~$2 OpenRouter).
