# Most related papers

The 20 papers closest to this project, grouped by what they're useful for. Local PDFs are named `NN_short_name.pdf`. The PDFs are not committed: arXiv licences don't all allow redistribution, so use the links. `docs/reading_list.md` has a longer list of about 60 papers.

Suggested reading order: 01 → 02 → 03 → 08 → 12 → 13 → 17 → 18 → the rest as needed.

## A. Environment and closest prior work

| # | Paper | arXiv | Why read it |
|---|---|---|---|
| 01 | SMACv2: An Improved Benchmark for Cooperative MARL | [2212.07489](https://arxiv.org/abs/2212.07489) | Our environment (`10gen_protoss` 5v5, random unit types and spawns). Defines the global state the MAPPO critic uses. |
| 02 | SMAC-Talk: A Natural Language Extension of SMAC for LLMs | [2606.04202](https://arxiv.org/abs/2606.04202) | The nearest neighbour: a text interface to SMAC with per-unit LLM control and a language channel. |
| 03 | COMPASS: Closed-Loop Vision-Language Planning for Multi-Agent Coordination | [2502.10148](https://arxiv.org/abs/2502.10148) | Decentralised VLM agents with structured communication on SMACv2. Reports 57% on Protoss 5v5, the number we compare against. |
| 04 | Large Language Models Play StarCraft II (TextStarCraft) | [2312.11865](https://arxiv.org/abs/2312.11865) | Text interface to SC2 and Chain-of-Summarization. Shows LLMs are weak at micro control. |
| 05 | LLM-PySC2 | [2411.05348](https://arxiv.org/abs/2411.05348) | Multi-agent LLM control of full SC2. Covers observation and action serialisation choices. |
| 06 | HLSMAC | [2509.12927](https://arxiv.org/abs/2509.12927) | Argues LLMs should play at a strategic, not micro, level. A counterpoint to our per-unit design. |
| 07 | LLM-Coordination | [2310.03903](https://arxiv.org/abs/2310.03903) | How to evaluate and analyse coordination between prompted LLM agents. |

## B. Multi-agent RL: the algorithms we use

| # | Paper | arXiv | Why read it |
|---|---|---|---|
| 08 | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO) | [2103.01955](https://arxiv.org/abs/2103.01955) | Shared actor plus centralised critic on the global state. This is our MAPPO route. |
| 09 | Generalized Advantage Estimation | [1506.02438](https://arxiv.org/abs/1506.02438) | GAE(γ, λ), used per agent on the joint timeline. |
| 10 | Proximal Policy Optimization | [1707.06347](https://arxiv.org/abs/1707.06347) | The clipped objective. We apply it per token with sampled old log-probs. |
| 11 | DeepSeekMath (GRPO) | [2402.03300](https://arxiv.org/abs/2402.03300) | Group-relative advantage with no critic. Our self-play GRPO route. |
| 19 | COMA: Counterfactual Multi-Agent Policy Gradients | [1705.08926](https://arxiv.org/abs/1705.08926) | The classic treatment of multi-agent credit assignment. Background for our per-unit credit term. |

## C. Multi-agent RL for LLMs

| # | Paper | arXiv | Why read it |
|---|---|---|---|
| 12 | LLM Collaboration with Multi-Agent RL (MAGRPO) | [2508.04652](https://arxiv.org/abs/2508.04652) | Dec-POMDP framing with group-relative advantages, centralised training and decentralised execution. The closest template for shared-weight units. |
| 13 | Stronger-MAS | [2510.11062](https://arxiv.org/abs/2510.11062) | Per-agent, per-turn grouping for multi-agent multi-turn GRPO. |
| 14 | MARFT: Multi-Agent Reinforcement Fine-Tuning | [2504.16129](https://arxiv.org/abs/2504.16129) | A formal map of LLM multi-agent RL fine-tuning, with open code. |
| 15 | MAPoRL | [2502.18439](https://arxiv.org/abs/2502.18439) | Joint post-co-training of several LLMs, with rewards that favour collaboration. |
| 16 | SPIRAL: Self-Play on Zero-Sum Games | [2506.24119](https://arxiv.org/abs/2506.24119) | Multi-turn self-play RL for LLMs, with a role-conditioned advantage. Relevant to our self-play route. |
| 20 | CCPO: Counterfactual Credit Policy Optimization | [2603.21563](https://arxiv.org/abs/2603.21563) | Counterfactual credit assignment for collaborating LLM agents. |

## D. Learning from a stronger model (our SFT baselines)

| # | Paper | arXiv | Why read it |
|---|---|---|---|
| 17 | Advantage-Weighted Regression | [1910.00177](https://arxiv.org/abs/1910.00177) | The AWR objective. With batch size 1 our version reduces to reward-filtered SFT (see the report). |
| 18 | DAgger | [1011.0686](https://arxiv.org/abs/1011.0686) | Interactive imitation: the student plays and the teacher (31B) labels the states it visits. |
