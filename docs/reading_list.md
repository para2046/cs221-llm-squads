# Reading list: small LLM squads in SMACv2

Built for the CS221 project in which Gemma 4 E2B/E4B controls a 5-unit SMACv2 squad, one model call per unit, with Gemma 4 31B as teacher.

**How this was checked (2026-11-16).** Every entry below was looked up through the arXiv API (`export.arxiv.org/api/query`, by `id_list` or title search). For each one, the exact title, first author, and first-posted date come from the API response. A venue is listed only when the arXiv comment or journal-ref field states it. When no venue is listed, treat the paper as a preprint. Many 2026 entries are recent preprints that have not been peer reviewed.

Legend: ★ = read first. [in report] = already in `report/references.bib`.

---

## ★ Read-first shortlist (11)

1. SMAC-Talk (2606.04202): the closest prior work to this project.
2. LLM Collaboration With Multi-Agent Reinforcement Learning, MAGRPO (2508.04652).
3. Stronger-MAS (2510.11062).
4. SPIRAL (2506.24119).
5. RAGEN (2504.20073).
6. DAPO (2503.14476).
7. Learning to Reason under Off-Policy Guidance, LUFFY (2504.14945).
8. On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents (2606.15912).
9. Training Language Models for Social Deduction with Multi-Agent RL (2502.06060).
10. Secret Collusion among AI Agents (2402.07510).
11. Why Do Multi-Agent LLM Systems Fail? (2503.13657).

---

## A. LLM agents in StarCraft II and other games (9)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| A1 ★ | SMAC-Talk: A Natural Language Extension of the StarCraft Multi-Agent Challenge for Large Language Models [in report] | 2606.04202 | Sol, 2026 | preprint | Nearest neighbour: a text interface to SMAC with decentralised LLM control and a language channel. Compare its observation and action format and its baselines with ours, and set our contribution apart from it. |
| A2 | LLM-PySC2: Starcraft II learning environment for Large Language Models [in report] | 2411.05348 | Z. Li, 2024 | preprint | Multi-agent LLM control of full SC2 through text and image observations. A reference for how to serialise observations and actions, and for its finding that LLMs struggle with micro. |
| A3 | HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making [in report] | 2509.12927 | Hong, 2025 | preprint | SMAC-style scenarios aimed at strategy rather than micro. Use it to argue which level of abstraction suits LLMs. |
| A4 | AVA: Attentive VLM Agent for Mastering StarCraft II | 2503.05383 | W. Ma, 2025 | ACL 2026 (per arXiv comment) | AVACraft benchmarks MARL methods (QMIX, MAPPO, ...) against zero-shot VLM/LLM agents on 21 micro/coordination scenarios. Gives a direct comparison protocol and baseline numbers. |
| A5 | SwarmBrain: Embodied agent for real-time strategy game StarCraft II via large language models | 2401.17749 | Shao, 2024 | preprint | Splits the controller into a slow LLM "overmind" and a fast reflexive layer. Relevant to latency, since each unit makes its own model call per step. |
| A6 | JEV-Star: Fast, Low-Cost StarCraft II Control with Language-Model Planning | 2609.27331 | W. Ma, 2026 | preprint | Reports cost and latency per game for LLM-in-the-loop SC2 control. Useful for our compute budget and cost/latency reporting. |
| A7 | SC-Phi2: A Fine-tuned Small Language Model for StarCraft II Macromanagement Tasks | 2409.18989 | Khan, 2024 | preprint | One of the few attempts to fine-tune a small LM (Phi-2) for SC2. Precedent for the small-model angle. |
| A8 | BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games | 2411.13543 | Paglieri, 2024 | ICLR 2025 | Shows how to evaluate LLM agents on long-horizon games with text observations, and how far small models fall short. Borrow its evaluation and reporting conventions. |
| A9 | lmgame-Bench: How Good are LLMs at Playing Games? | 2505.15146 | Hu, 2025 | preprint | Gives game harness design (perception and memory scaffolds) and finds that RL on games transfers. Use it to justify harness choices. |

## B. Multi-agent LLM RL (8)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| B1 ★ | LLM Collaboration With Multi-Agent Reinforcement Learning (MAGRPO) [in report] | 2508.04652 | S. Liu, 2025 | preprint | Treats LLM collaboration as a Dec-POMDP and uses group-relative advantages with centralised training and decentralised execution. Closest algorithmic template for shared-weight units. |
| B2 ★ | Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs [in report] | 2510.11062 | Zhao, 2025 | preprint | Grouping and advantage estimation per agent and per turn for multi-agent multi-turn GRPO, plus a role-aware or shared-policy system design. Copy its grouping logic. |
| B3 | MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning [in report] | 2502.18439 | Park, 2025 | ACL (per arXiv comment) | Co-training several LLMs with a verifier-shaped reward that rewards collaboration. Shows that joint training beats training each agent alone. |
| B4 | MARFT: Multi-Agent Reinforcement Fine-Tuning [in report] | 2504.16129 | Liao, 2025 | DAI 2026 (extended version) | Formalises LLM multi-agent RL fine-tuning (Flex-POMDP) and gives open code. A good conceptual map and source of implementation details. |
| B5 | ProST: Progressive Sub-task Training for Pareto-Optimal Multi-agent Systems Using Small Language Models [in report] | 2509.04508 | Bijoy, 2025 | preprint | Small LMs in multi-agent systems trained with a curriculum. Supports a curriculum over SMACv2 difficulty and unit count. |
| B6 | Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning (CORY) | 2410.06101 | H. Ma, 2024 | NeurIPS 2024 | Duplicates one LLM into cooperating pioneer and observer copies that swap roles. Relevant to how much shared-weight agents benefit from role asymmetry. |
| B7 | MALT: Improving Reasoning with Multi-Agent LLM Training | 2412.01928 | Motwani, 2024 | COLM 2025 | Assigns credit across a generator, verifier, and refiner chain using value-iteration-style propagation. A pattern for credit assignment across unit roles. |
| B8 | MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs | 2510.15414 | Yuan, 2025 | preprint | Multi-turn multi-agent self-play RL that tackles long-horizon credit assignment and advantage estimation per agent. Directly relevant to items (1) and (3). |

## 1. Live on-policy RL objectives for multi-turn agents (9)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 1.1 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO) | 2402.03300 | Shao, 2024 | preprint | Introduces GRPO, which uses a group baseline instead of a critic. Our baseline objective, and cheap enough to run on ≤4B models. |
| 1.2 ★ | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | 2503.14476 | Yu, 2025 | preprint | Clip-higher, dynamic sampling (drop groups where every outcome is the same), token-level loss, and overlong shaping. Dynamic sampling matters a lot when most SMAC episodes are all-win or all-loss. |
| 1.3 | Group Sequence Policy Optimization (GSPO) | 2507.18071 | Zheng, 2025 | preprint | Importance ratios at sequence level are more stable than token-level GRPO ratios, especially for long multi-turn outputs. An ablation to try. |
| 1.4 | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | 2503.20783 | Z. Liu, 2025 | preprint | Shows GRPO's length and std normalisation biases and gives an unbiased fix. Use it to avoid reward-irrelevant length drift in unit messages. |
| 1.5 ★ | RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning | 2504.20073 | Z. Wang, 2025 | preprint | StarPO for multi-turn agent RL. Documents an "echo trap" collapse and the stabilisers that prevent it. Expect the same failure modes in SMAC. |
| 1.6 | Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) | 2505.10978 | Feng, 2025 | NeurIPS 2025 | Groups episodes and also groups anchor states that repeat across rollouts, giving credit per step without a critic. Fits SMAC, where similar states recur. |
| 1.7 | ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL | 2402.19446 | Zhou, 2024 | preprint | Splits learning into a high-level critic over turns and a low-level token policy. The main alternative to critic-free GRPO when episodes are long. |
| 1.8 | True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning (TWOSOME) | 2401.14151 | Tan, 2024 | ICLR 2024 | PPO on the probabilities an LLM assigns to valid actions, with normalisation for action length. Directly applicable to scoring SMAC's discrete action menu. |
| 1.9 | SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks | 2503.15478 | Zhou, 2025 | preprint | Uses an asymmetric critic that sees privileged training-time information for step-level credit. Maps naturally to a centralised critic that sees the global SMAC state. |

## 2. Learning from teacher trajectories (BC, offline RL, off-policy guidance, on-policy distillation) (7)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 2.1 ★ | Learning to Reason under Off-Policy Guidance (LUFFY) | 2504.14945 | Yan, 2025 | preprint | Mixes teacher (off-policy) rollouts into GRPO groups and uses regularised importance sampling to avoid entropy collapse. The recipe for injecting Gemma-31B trajectories into live RL. |
| 2.2 ★ | On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents | 2606.15912 | G. Li, 2026 | preprint | On-policy distillation fails in multi-turn agents because student errors push states out of the teacher's distribution. Proposes turn-level guidance as a fix. This is our DAgger-with-LLM-teacher setting. |
| 2.3 | Stochastic Teacher Intervention for Agentic On-Policy Distillation | 2610.10878 | J. Liu, 2026 | preprint | Lets the teacher take over some turns at random so that student rollouts stay where the teacher's labels are reliable. Close to DAgger with a mixing parameter β. |
| 2.4 | Beyond the Teacher: Understanding and Preserving Distillation Overshoot in Agentic On-Policy Distillation | 2610.25034 | Zhong, 2026 | preprint | A weaker student can briefly beat its teacher during on-policy distillation and then regress. Tells us to checkpoint and early-stop when the student is compared with the 31B teacher. |
| 2.5 | Offline Reinforcement Learning for LLM Multi-Step Reasoning (OREO) | 2412.16145 | H. Wang, 2024 | preprint | Learns a value function and policy jointly from fixed trajectories (soft Bellman). An offline-RL alternative to advantage-weighted BC on teacher logs. |
| 2.6 | Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions (ReLIFT) | 2506.07527 | L. Ma, 2025 | ICLR 2026 | Interleaves RL with SFT on teacher demonstrations, but only for cases RL fails on. A simple schedule for mixing BC and RL on hard SMACv2 seeds. |
| 2.7 | DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning | 2406.11896 | Bai, 2024 | preprint | Moves from offline to online RL with AWR-style filtering and a learned instruction-level value function in a stochastic interactive environment. A template for BC, then offline RL, then online RL. |

## 3. Model-vs-model play, self-play, and populations (5)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 3.1 ★ | SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning [in report] | 2506.24119 | B. Liu, 2025 | ICLR 2026 | Multi-turn self-play RL for LLMs in zero-sum games, with role-conditioned advantage estimation. The reference design for small-vs-large and self-play matches. |
| 3.2 | TextArena | 2504.11442 | Guertler, 2025 | preprint | Text-game environment suite with online model-vs-model play and TrueSkill-style ratings. Use its rating methodology for our league or Elo tables. |
| 3.3 | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN) | 2401.01335 | Z. Chen, 2024 | ICML 2024 | Self-play against the model's own earlier iterate, set up as a two-player game. Background for opponent sampling from past checkpoints. |
| 3.4 | Self-playing Adversarial Language Game Enhances LLM Reasoning (SPAG) | 2404.10642 | Cheng, 2024 | NeurIPS 2024 | Adversarial self-play in a language game improves general reasoning. Supports the claim that competitive play produces transferable skill. |
| 3.5 | Richelieu: Self-Evolving LLM-Based Agents for AI Diplomacy | 2407.06813 | Guan, 2024 | NeurIPS 2024 | LLM agents improve through self-play in a mixed-motive, communication-heavy game. Bridges item (3) with items (4) and (5). |

## 4. Inter-agent communication and emergent communication (5)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 4.1 ★ | Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning | 2502.06060 | Sarkar, 2025 | AAMAS 2025 | Splits communication into listening and speaking, each with its own dense reward, and trains both by multi-agent RL without human data. The most direct recipe for training our unit message channel. |
| 4.2 | Language Grounded Multi-agent Reinforcement Learning with Human-interpretable Communication (LangGround) | 2409.17348 | H. Li, 2024 | NeurIPS 2024 | Grounds a learned MARL communication space in LLM-generated language. Relevant to keeping emergent messages interpretable, and to monitoring them. |
| 4.3 | Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System | 2410.08115 | W. Chen, 2024 | preprint | Trains LLM agents to communicate more briefly with no loss in task reward (iterative SFT/DPO with a token-cost reward). Pattern for a message-length penalty. |
| 4.4 | Communicating Activations Between Language Model Agents | 2501.14082 | Ramesh, 2025 | ICML 2025 | Shows that a non-text (activation) channel can outperform natural language. A useful contrast, and a warning for collusion monitoring. |
| 4.5 | Embodied LLM Agents Learn to Cooperate in Organized Teams | 2403.12482 | Guo, 2024 | preprint | Communication structure such as a leader or hierarchy changes how efficiently embodied LLM teams work. Consider a squad-leader message topology. |

## 5. Collusion, covert channels, and monitoring (11)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 5.1 ★ | Secret Collusion among AI Agents: Multi-Agent Deception via Steganography [in report] | 2402.07510 | Motwani, 2024 | preprint | Defines secret collusion and steganographic capability evaluations. The conceptual basis for item (5). |
| 5.2 | Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems [in report] | 2602.15198 | Nakamura, 2026 | preprint | An auditing framework for collusion in cooperative multi-agent systems. Borrow its metrics for our collusion probe. |
| 5.3 | Emergent Collusion in Long-Horizon LLM Agent Interaction [in report] | 2609.24967 | Shi, 2026 | preprint | Collusion appears in 94% of trajectories once reward is reachable only by breaking a protocol. Shows that collusion can emerge over repeated episodes without explicit instruction. |
| 5.4 | Voluntary Collusion with Secret Tools in Competing LLM Agents [in report] | 2605.27593 | Zeng, 2026 | preprint | Safety-aligned agents take secret collusion tools when doing so is advantageous, across scales from 7B up. Close to our "secret incentive plus shared channel" design. |
| 5.5 | Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs | 2410.03768 | Mathew, 2024 | IJCNLP-AACL 2025 (oral) | Steganography emerges under optimisation pressure (RL and in-context) and evades paraphrase defences. Key evidence that RL on the channel could produce covert messages. |
| 5.6 | Early Signs of Steganographic Capabilities in Frontier LLMs | 2507.02737 | Zolkowski, 2025 | preprint | Measures encoded-message and encoded-reasoning ability in current models. Calibrates whether Gemma-31B is plausibly capable of covert signalling. |
| 5.7 | Subversion via Focal Points: Investigating Collusion in LLM Monitoring | 2507.03010 | Järviniemi, 2025 | preprint | Collusion without any communication, through Schelling points. Our probe needs a no-channel control condition for this reason. |
| 5.8 | Algorithmic Collusion by Large Language Models | 2404.00806 | Fish, 2024 | EC 2026 (per arXiv comment) | LLM pricing agents collude tacitly with no instruction. The canonical economics-style collusion result, and a template for collusion metrics. |
| 5.9 | Detecting Multi-Agent Collusion Through Multi-Agent Interpretability | 2604.01151 | Rose, 2026 | preprint | Combines activation probes from each agent into joint hypothesis tests that detect a colluding subgroup. A monitoring option, since we control the open weights. |
| 5.10 | What Do Collusion Probes Read? Three Pitfalls in Evaluating Multi-Agent Collusion Monitors | 2611.10620 | Mohsin, 2026 | preprint | Probes mostly detect that an agent was told to collude, not that it decided to. Our evaluation must include agents that receive the instruction and refuse it. |
| 5.11 | SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents [in report] | 2506.15740 | Kutasov, 2025 | preprint | Paired main and side tasks with an LLM monitor, scored by AUC. A methodology template for monitoring hidden objectives. |

## 6. Coordination benchmarks, multi-agent failure analyses, cooperative AI safety (6)

| # | Paper | arXiv | First author, year | Venue | Why it matters here |
|---|---|---|---|---|---|
| 6.1 ★ | Why Do Multi-Agent LLM Systems Fail? | 2503.13657 | Cemri, 2025 | preprint | MAST failure taxonomy (specification, inter-agent misalignment, verification). Use it to code qualitative failures in our unit transcripts. |
| 6.2 | Reward and Behavioral Measures of LLM Coordination in Dining Philosophers [in report] | 2606.07845 | Hasan, 2026 | preprint | High reward can hide a coordination failure (reward with zero meals). Report behavioural metrics such as focus fire and survival, not only win rate. |
| 6.3 | Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents (GovSim) [in report] | 2404.16698 | Piatti, 2024 | NeurIPS 2024 | A mixed-motive LLM society where most models fail to sustain cooperation and communication helps. Related evidence for items (4) and (5). |
| 6.4 | MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents | 2503.01935 | Zhu, 2025 | preprint | Milestone-based metrics for collaboration and competition, plus a comparison of coordination topologies. Use its metric design and topology ablations. |
| 6.5 | Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents | 2502.20073 | Sun, 2025 | EMNLP 2025 | Process-level collaboration metrics in a real-time cooperative game. Close in spirit to evaluating unit coordination. |
| 6.6 | Multi-Agent Risks from Advanced AI | 2502.14143 | Hammond, 2025 | Cooperative AI Foundation Tech. Report #1 | Taxonomy of miscoordination, conflict, and collusion risks. Frames the collusion probe within cooperative AI safety. |

---

## Background: pre-2024 classics (5)

| Paper | arXiv | First author, year | Why |
|---|---|---|---|
| SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning [in report] | 2212.07489 | Ellis, 2022 | The environment we use: procedurally generated unit types and start positions, plus a stochasticity critique of SMACv1. |
| Large Language Models Play StarCraft II: Benchmarks and A Chain of Summarization Approach (TextStarCraft II) [in report] | 2312.11865 | W. Ma, 2023 | The first text interface for LLMs in SC2, and the origin of the chain-of-summarization observation compression. |
| Proximal Policy Optimization Algorithms [in report] | 1707.06347 | Schulman, 2017 | The clipped-ratio objective that GRPO, DAPO, and GSPO all modify. |
| A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) [in report] | 1011.0686 | Ross, 2010 (AISTATS 2011) | Explains why pure BC on teacher trajectories compounds errors, and why to query the teacher on student states. |
| On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) | 2306.13649 | Agarwal, 2023 (ICLR 2024) | The original LLM on-policy distillation objective that the 2026 agentic on-policy distillation papers in section 2 extend. |

---

### Notes
- Count: 60 main entries (A 9, B 8, §1 9, §2 7, §3 5, §4 5, §5 11, §6 6) plus 5 background papers. All sixteen 2024–2026 entries from `references.bib` are included and marked [in report].
- Pre-2024 bib entries that are not listed above (AWR, GAE, MAPPO, QMIX, Other-Play, LoRA, PySC2, SMACv1, HoK Arena, LLM-Coordination) were left out to respect the limit of 5 background papers. They stay in the report bibliography.
