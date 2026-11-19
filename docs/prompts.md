# Prompts and the communication channel

This page shows exactly what each model sees and returns. Every example below is copied from our game logs.

## Who gets which prompt

- **One system prompt for everyone.** The untrained 4B, the trained 4B (SFT, DAgger, MAPPO, GRPO) and the 31B teacher all get the same system prompt. The only thing that changes it is the channel (C0 / C1 / C2): a block of text is appended for C1, and another for C2.
- **No per-role system prompt.** A unit's *role* (stalker, zealot or colossus) appears only in its per-step observation ("You are agent a1, a stalker ..."). The system prompt lists stats for all three unit types.
- **No tactics anywhere.** The prompt contains rules and unit facts only ("TACTICS: You decide your own tactics."). Our win/loss analysis is never put into prompts.
- **The reward is not in the prompt.** Agents see only the game's default SMAC score ("Team score so far"). Shaped rewards are used for training only.
- **One model call per unit per step.** Each of the 5 units is a separate call with its own prompt. The 5 replies together form the joint action. Calls never see each other's replies except through the channel below.
- Source: `smac_llm/text_env.py` (`SYSTEM_PROMPT`, `UNIFIED_PROMPT`, `UNIFIED_INFO`, `UNIFIED_MSG`), `smac_llm/game.py::system_prompt`. Protocol: `docs/unified_protocol.md`.

## 1. System prompt

### C0: no channel (base text, shared by all conditions)

```text
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

### C1: base text + team reports

```text
TEAM REPORTS
Every step you receive one JSON object per living teammate: its type, position, hp, shield, weapon cooldown and the enemies it can see (id, type, position, hp, shield). Your own report is sent automatically.
```

### C2: base text + team reports + free message

```text
TEAM MESSAGES
You may add a "msg" field (at most 8 words) to your reply; your teammates see it in your next report. Format: {"action": "<one legal action>", "msg": "<at most 8 words>"}
```

## 2. Per-step user prompt: the same unit in C0, C1 and C2

These are scenario 100004, step 3, agent a1 (a stalker), untrained 4B. The scenario seed fixes the unit types and our start positions, but positions drift as the games diverge, so the three moments are not identical.

### C0: own view only

The unit sees only what is within its own sight range (10). Here no enemy is visible yet, so it walks.

```text
Step 3. You are agent a1, a stalker at (17.5, 20.0), hp 80/80, shield 80/80, weapon cooldown 0, shoot range 6, sight range 10.
Visible teammates:
  a0 stalker at (15.2, 18.7), dist 2.7, hp 80/80, shield 80/80
  a2 zealot at (17.3, 15.7), dist 4.3, hp 100/100, shield 50/50
  a3 zealot at (18.2, 16.2), dist 3.9, hp 100/100, shield 50/50
  a4 zealot at (20.3, 16.0), dist 4.9, hp 100/100, shield 50/50
Visible enemies: none
Units alive: your team 5, enemy 5.
Your last steps (oldest first):
  step 0: move_north ->; you 160 -> 160
  step 1: move_east ->; you 160 -> 160
  step 2: move_north ->; you 160 -> 160
Team score so far: 0.0 / 20
Legal actions: stop, move_north, move_south, move_east, move_west
```
Reply: `{"action": "move_north"}`

### C1: + one JSON "team report" per living teammate

Each report carries the teammate's state and the enemies *it* can see. A unit can therefore know about enemies outside its own sight.

```text
Step 3. You are agent a1, a stalker at (16.0, 17.1), hp 80/80, shield 80/80, weapon cooldown 9, shoot range 6, sight range 10.
Visible teammates:
  a0 stalker at (16.4, 15.8), dist 1.3, hp 80/80, shield 80/80
  a2 zealot at (18.2, 13.9), dist 3.9, hp 100/100, shield 37/50
  a3 zealot at (18.1, 14.9), dist 3.0, hp 100/100, shield 50/50
  a4 zealot at (18.2, 16.1), dist 2.4, hp 100/100, shield 50/50
Visible enemies:
  e2 stalker at (19.1, 12.9), dist 5.2, hp 80/80, shield 44/80 [in range]
  e4 zealot at (11.0, 11.0), dist 7.9, hp 100/100, shield 50/50
Units alive: your team 5, enemy 5.
Team reports:
  {"from":"a0","type":"stalker","pos":[16.4,15.8],"hp":80,"shield":80,"cooldown":10,"sees":[{"id":"e2","type":"stalker","pos":[19.1,12.9],"hp":80,"shield":44},{"id":"e4","type":"zealot","pos":[11.0,11.0],"hp":100,"shield":50}]}
  {"from":"a2","type":"zealot","pos":[18.2,13.9],"hp":100,"shield":37,"cooldown":0,"sees":[{"id":"e2","type":"stalker","pos":[19.1,12.9],"hp":80,"shield":44},{"id":"e4","type":"zealot","pos":[11.0,11.0],"hp":100,"shield":50}]}
  {"from":"a3","type":"zealot","pos":[18.1,14.9],"hp":100,"shield":50,"cooldown":0,"sees":[{"id":"e2","type":"stalker","pos":[19.1,12.9],"hp":80,"shield":44},{"id":"e4","type":"zealot","pos":[11.0,11.0],"hp":100,"shield":50}]}
  {"from":"a4","type":"zealot","pos":[18.2,16.1],"hp":100,"shield":50,"cooldown":0,"sees":[{"id":"e2","type":"stalker","pos":[19.1,12.9],"hp":80,"shield":44},{"id":"e4","type":"zealot","pos":[11.0,11.0],"hp":100,"shield":50}]}
Your last steps (oldest first):
  step 0: attack_e2 -> e2 hp+shield 160 -> 160; you 160 -> 160
  step 1: attack_e2 -> e2 hp+shield 160 -> 124; you 160 -> 160
  step 2: attack_e2 -> e2 hp+shield 124 -> 124; you 160 -> 160
Team score so far: 0.7 / 20
Legal actions: stop, move_north, move_south, move_east, move_west, attack_e2
```
Reply: `{"action": "attack_e2"}`

### C2: + a free message (≤ 8 words) inside each report

The teammate's last `msg` appears in its report on the next step.

```text
Step 3. You are agent a1, a stalker at (17.5, 17.1), hp 80/80, shield 62/80, weapon cooldown 17, shoot range 6, sight range 10.
Visible teammates:
  a0 stalker at (17.2, 15.8), dist 1.3, hp 80/80, shield 80/80
  a2 zealot at (19.4, 15.1), dist 2.8, hp 100/100, shield 50/50
  a3 zealot at (19.4, 16.0), dist 2.2, hp 100/100, shield 50/50
  a4 zealot at (20.3, 16.0), dist 3.0, hp 100/100, shield 50/50
Visible enemies:
  e0 zealot at (21.2, 20.4), dist 4.9, hp 100/100, shield 50/50 [in range]
  e1 stalker at (21.7, 22.8), dist 7.1, hp 80/80, shield 80/80
  e3 zealot at (20.2, 19.4), dist 3.6, hp 100/100, shield 37/50 [in range]
  e4 zealot at (20.2, 20.3), dist 4.3, hp 100/100, shield 50/50 [in range]
Units alive: your team 5, enemy 5.
Team reports:
  {"from":"a0","type":"stalker","pos":[17.2,15.8],"hp":80,"shield":80,"cooldown":23,"sees":[{"id":"e0","type":"zealot","pos":[21.2,20.4],"hp":100,"shield":50},{"id":"e1","type":"stalker","pos":[21.7,22.8],"hp":80,"shield":80},{"id":"e3","type":"zealot","pos":[20.2,19.4],"hp":100,"shield":37},{"id":"e4","type":"zealot","pos":[20.2,20.3],"hp":100,"shield":50}]}
  {"from":"a2","type":"zealot","pos":[19.4,15.1],"hp":100,"shield":50,"cooldown":0,"sees":[{"id":"e0","type":"zealot","pos":[21.2,20.4],"hp":100,"shield":50},{"id":"e1","type":"stalker","pos":[21.7,22.8],"hp":80,"shield":80},{"id":"e2","type":"stalker","pos":[24.0,8.3],"hp":80,"shield":80},{"id":"e3","type":"zealot","pos":[20.2,19.4],"hp":100,"shield":37},{"id":"e4","type":"zealot","pos":[20.2,20.3],"hp":100,"shield":50}],"msg":"Move towards enemy line"}
  {"from":"a3","type":"zealot","pos":[19.4,16.0],"hp":100,"shield":50,"cooldown":0,"sees":[{"id":"e0","type":"zealot","pos":[21.2,20.4],"hp":100,"shield":50},{"id":"e1","type":"stalker","pos":[21.7,22.8],"hp":80,"shield":80},{"id":"e3","type":"zealot","pos":[20.2,19.4],"hp":100,"shield":37},{"id":"e4","type":"zealot","pos":[20.2,20.3],"hp":100,"shield":50}],"msg":"Advance to engage"}
  {"from":"a4","type":"zealot","pos":[20.3,16.0],"hp":100,"shield":50,"cooldown":0,"sees":[{"id":"e0","type":"zealot","pos":[21.2,20.4],"hp":100,"shield":50},{"id":"e1","type":"stalker","pos":[21.7,22.8],"hp":80,"shield":80},{"id":"e2","type":"stalker","pos":[24.0,8.3],"hp":80,"shield":80},{"id":"e3","type":"zealot","pos":[20.2,19.4],"hp":100,"shield":37},{"id":"e4","type":"zealot","pos":[20.2,20.3],"hp":100,"shield":50}]}
Your last steps (oldest first):
  step 0: move_east ->; you 160 -> 160
  step 1: attack_e3 -> e3 hp+shield 150 -> 150; you 160 -> 160
  step 2: attack_e3 -> e3 hp+shield 150 -> 137; you 160 -> 142
Team score so far: 0.3 / 20
Legal actions: stop, move_north, move_south, move_east, move_west, attack_e0, attack_e3, attack_e4
```
Reply: `{"action": "attack_e3"}`. The untrained 4B often ignores the channel: here it neither writes a message nor follows "Advance to engage".

## 3. A colossus prompt in C2 (trained 4B, SFT on 31B games)

Scenario 100046, step 7, agent a0. This is the game shown in the demo. The history block shows how its own target, e3, was worn down and killed.

```text
Step 7. You are agent a0, a colossus at (16.0, 16.0), hp 200/200, shield 150/150, weapon cooldown 16, shoot range 7, sight range 10.
Visible teammates:
  a1 stalker at (16.0, 17.1), dist 1.1, hp 80/80, shield 80/80
  a2 stalker at (17.1, 16.0), dist 1.1, hp 80/80, shield 44/80
  a3 zealot at (17.4, 14.5), dist 2.0, hp 100/100, shield 50/50
  a4 zealot at (16.0, 16.0), dist 0.0, hp 100/100, shield 50/50
Visible enemies:
  e2 stalker at (9.8, 22.7), dist 9.1, hp 80/80, shield 80/80
Units alive: your team 5, enemy 4.
Team reports:
  {"from":"a1","type":"stalker","pos":[16.0,17.1],"hp":80,"shield":80,"cooldown":5,"sees":[{"id":"e2","type":"stalker","pos":[9.8,22.7],"hp":80,"shield":80}],"msg":"Finishing e3"}
  {"from":"a2","type":"stalker","pos":[17.1,16.0],"hp":80,"shield":44,"cooldown":6,"sees":[{"id":"e2","type":"stalker","pos":[9.8,22.7],"hp":80,"shield":80}],"msg":"Finishing off e3"}
  {"from":"a3","type":"zealot","pos":[17.4,14.5],"hp":100,"shield":50,"cooldown":14,"sees":[],"msg":"Finishing e3"}
  {"from":"a4","type":"zealot","pos":[16.0,16.0],"hp":100,"shield":50,"cooldown":0,"sees":[],"msg":"Closing in on e3"}
Your last steps (oldest first):
  step 2: attack_e3 -> e3 hp+shield 96 -> 88; you 350 -> 350
  step 3: attack_e3 -> e3 hp+shield 88 -> 80; you 350 -> 350
  step 4: attack_e3 -> e3 hp+shield 80 -> 38; you 350 -> 350
  step 5: attack_e3 -> e3 hp+shield 38 -> 21; you 350 -> 350
  step 6: attack_e3 -> e3 killed; you 350 -> 350
Team score so far: 3.4 / 20
Legal actions: stop, move_north, move_south, move_east, move_west
```
Reply: `{"action": "move_north", "msg": "Moving to engage e2"}`

## 4. How the channel is used: examples

**Focus fire called through messages** (same game, trained 4B). At step 6 all four units that are in range attack e3 and say so. e3 dies that step. At step 7 every unit switches to the next visible enemy, e2, and announces it:

| step | agent | reply |
|---|---|---|
| 6 | a0 | `{"action": "attack_e3", "msg": "Finishing e3"}` |
| 6 | a1 | `{"action": "attack_e3", "msg": "Finishing e3"}` |
| 6 | a2 | `{"action": "attack_e3", "msg": "Finishing off e3"}` |
| 6 | a3 | `{"action": "attack_e3", "msg": "Finishing e3"}` |
| 6 | a4 | `{"action": "move_east", "msg": "Closing in on e3"}` |
| 7 | a0 | `{"action": "move_north", "msg": "Moving to engage e2"}` |
| 7 | a1 | `{"action": "move_north", "msg": "Closing in on e2"}` |
| 7 | a2 | `{"action": "move_north", "msg": "Closing in on e2"}` |
| 7 | a3 | `{"action": "move_north", "msg": "Moving to e2"}` |
| 7 | a4 | `{"action": "move_north", "msg": "Moving to e2"}` |

**Gemma 4 31B (teacher), 208 games in C2: what it writes** (29,117 messages, 2,661 distinct; counts are messages matching each pattern, and the patterns overlap):

| Kind | Count | Typical messages |
|---|---|---|
| Name a target (focus / finish / kill) | 16,505 | "Focusing e1", "Finishing e3" |
| Search / move to join | 2,465 | "Moving north to find enemies", "Moving east to support a4", "Moving south to join team" |
| Status / warning | 888 | "Retreating, low health", "Retreating to recover shields", "heading to help a2" |
| Plan or regroup | 680 (mostly "retreating") | real regroup plans are under 1% of messages |

**Following a call** (31B, `runs/d31b_c2/w1.jsonl`, episode 0, step 31). a0 says "Focusing e1". On the next step a1 and a2, which had been doing something else (kiting, attacking e3), both attack e1:

| agent | step t: action, msg | step t+1: action |
|---|---|---|
| a0 | attack_e1, "Focusing e1" | move_south |
| a1 | move_north, "Kiting zealot e1" | **attack_e1** |
| a2 | attack_e3, "Finish e3" | **attack_e1** |
| a3 | attack_e3, "Finishing e3" | attack_e4 |

Measured over all 31B C2 games: a teammate that can hit the enemy named in a message attacks it 70% of the time, against 45% by chance (see the report, Q1).

## 5. Training targets

- **SFT on 31B games (and DAgger).** The target is the 31B's full reply (`action` + `msg`) for the same prompt format. DAgger: the 4B plays and the 31B labels the states the 4B visits.
- **MAPPO / GRPO.** The same prompt. The sampled reply tokens and their log-probs are stored and reused as the PPO "old" policy. The shaped reward is never shown to the model.
