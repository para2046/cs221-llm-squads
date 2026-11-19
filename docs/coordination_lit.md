# How multi-agent systems actually coordinate: MARL communication, CTDE, and LLM teams
# 多智能体如何真正实现协同：MARL 通信、CTDE 与 LLM 团队

Companion to `docs/reading_list.md`, focused on one question: **what mechanisms make agents coordinate (especially focus fire), and does the communication channel actually matter?** Context: Gemma 4 E2B/E4B units in SMACv2, one call per unit per step, shared weights, partial observability, optional `say` message (<=12 words) read by teammates next step. Experiments: (a) comm vs no-comm, (b) 31B vs built-in AI, (c) distill 4B from 31B wins, (d) 4B self-play RL with LoRA.

本文是 `docs/reading_list.md` 的补充，只关注一个问题：**什么机制让智能体真正协同（特别是集火），通信信道是否真的起作用？**

**Verification (2026-11-16).** Every arXiv id below was checked against the arXiv API (`export.arxiv.org/api/query`) for title, first author and date. Venues are given only when the arXiv comment field or the publisher page states them; otherwise treat the paper as a preprint. MAIC has no arXiv version; it was checked on the AAAI proceedings page. The "how it works" summaries for papers from 2023 onward come from the abstracts. Summaries of classic papers also draw on well-known content. Where a number is quoted, it is the number in the abstract.

**核实说明。** 所有 arXiv 编号均已通过 arXiv API 核对标题、第一作者与日期；会议信息仅在 arXiv 注释或出版方页面注明时给出。MAIC 无 arXiv 版本，已在 AAAI 论文集页面核实。

Cross-references like **[RL B1]** point to entries in `docs/reading_list.md` and are not repeated in full here.
形如 **[RL B1]** 的标注指向 `reading_list.md` 中已有条目，此处不再重复。

---

## 1. Classic learned communication in MARL / 经典 MARL 学习式通信

### 1.1 CommNet: Learning Multiagent Communication with Backpropagation
Sukhbaatar et al., 2016, NIPS 2016. https://arxiv.org/abs/1605.07736

**How:** At each communication step, every agent receives the mean of all other agents' hidden states as a continuous "message". The message is fed into the next layer, so the whole team is one differentiable network trained end to end with the task reward. Nothing is discrete or hand-designed: the protocol is just whatever gradients find useful.
**Result:** Outperforms no-communication and fully-connected baselines on traffic-junction and combat tasks, and the learned messages could be partially interpreted.
**中文：** 每个智能体把其他智能体隐藏状态的均值作为连续"消息"输入下一层，整个团队端到端反向传播训练；在交通路口和战斗任务上优于无通信基线。
**Relevance to our project / 对我们项目的意义:** This is the baseline argument that pooling teammates' state helps under partial observability. Our text `say` channel is a discrete, lossy version of the same idea. / 证明"汇总队友信息"在部分可观测下有用；我们的文本 `say` 是其离散、有损版本。

### 1.2 DIAL / RIAL: Learning to Communicate with Deep Multi-Agent Reinforcement Learning
Foerster et al., 2016, NIPS 2016. https://arxiv.org/abs/1605.06676

**How:** RIAL treats the message as just another discrete action learned by Q-learning. DIAL instead passes real-valued messages during centralised training and backpropagates the receiver's error gradient through the channel into the sender. At execution time it discretises the messages. Parameter sharing across agents is a key ingredient.
**Result:** DIAL learns protocols (switch riddle, MNIST communication games) that RIAL learns slowly or not at all. The reason is that a message sent as an RL action gets credit only through noisy reward.
**中文：** RIAL 把消息当作离散动作用 Q 学习训练；DIAL 在集中训练时让接收方梯度穿过信道回传给发送方，执行时再离散化。DIAL 能学会 RIAL 学不会的协议。
**Relevance:** This is the core warning for experiment (d). A text message trained only by sparse win/loss reward (RIAL-style) gets very weak credit. We need a denser signal for speaking, such as influence- or listener-based rewards (see 2.2 and 6.1). / 实验 (d) 的核心警示：仅靠稀疏胜负奖励训练文本消息，信用分配极弱，需要更稠密的"说话"信号。

### 1.3 TarMAC: Targeted Multi-Agent Communication
Das et al., 2018, ICML 2019. https://arxiv.org/abs/1810.11187

**How:** Each message has a signature (key) and a value. Receivers compute attention over senders with a query from their own state, so each agent effectively chooses whom to listen to. Multiple rounds of communication per step are possible. Everything is learned end to end from team reward.
**Result:** Beats CommNet-style broadcasting on SHAPES, traffic junction and House3D navigation. The learned attention concentrates on the senders that matter.
**中文：** 消息由"签名(键)+内容(值)"组成，接收方用注意力选择听谁；可多轮通信；在多个环境优于广播式通信。
**Relevance:** Suggests structured messages that name a recipient or target (e.g. `@all focus e3`) instead of free chat. Messages addressed to the units that can reach the target are what matter for focus fire. / 提示使用带指向性的结构化消息（如 `@all focus e3`），而非自由闲聊。

### 1.4 IC3Net: Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks
Singh et al., 2018, ICLR 2019. https://arxiv.org/abs/1812.09755

**How:** It extends CommNet with a learned binary gate per agent that decides whether to communicate at this step. Training uses individual rewards, so the method also works in mixed or competitive settings.
**Result:** Learns when to stay silent, scales to more agents, and outperforms CommNet in cooperative and competitive variants of predator-prey and traffic junction.
**中文：** 在 CommNet 上加入"是否发言"的可学习门控，使用个体奖励，可扩展到更多智能体和竞争场景。
**Relevance:** Our units already decide whether to speak. IC3Net motivates logging the speak rate, and testing whether "speak only when you change target or see a new enemy" outperforms always speaking. / 支持记录发言率，并测试"仅在换目标或发现新敌人时发言"是否优于每步发言。

### 1.5 NDQ: Learning Nearly Decomposable Value Functions via Communication Minimization
T. Wang et al., 2019, ICLR 2020. https://arxiv.org/abs/1910.05366

**How:** This is a SMAC method built on value factorisation (QMIX-style). Agents send messages that are trained with an information-theoretic objective. The objective maximises the mutual information between a message and the receiver's action selection, and minimises the message's entropy, so most messages can be dropped.
**Result:** On SMAC maps that need coordination under partial observability, it outperforms QMIX and other baselines. The authors report that a large fraction of messages can be cut without hurting performance.
**中文：** SMAC 上的值分解+通信方法：训练消息使其与接收方动作的互信息最大、熵最小，大部分消息可被删除而不降性能。
**Relevance:** An objective for a "good message": it should change what teammates do. This is a direct template for a speaker reward (see Idea 3 at the end) and for a message-length or entropy penalty. / "好消息"应当改变队友的行动；可直接作为说话奖励和消息长度惩罚的模板。

### 1.6 TMC: Succinct and Robust Multi-Agent Communication with Temporal Message Control
S. Q. Zhang et al., 2020, NeurIPS 2020. https://arxiv.org/abs/2010.14391

**How:** Agents send a new message only when it differs enough from the last one, and receivers buffer old messages. A regulariser encourages temporally smooth messages. This makes communication sparse and robust to message loss.
**Result:** On SMAC, it greatly reduces communication volume at similar win rates and is robust under lossy channels.
**中文：** 仅在消息变化足够大时才发送，接收方缓存旧消息；在 SMAC 上大幅减少通信量且对丢包鲁棒。
**Relevance:** Our next-step chat is already delayed by one step. TMC suggests keeping a short message history ("last message from each ally") in the prompt rather than only the latest step. / 建议在提示中保留每位队友的最近一条消息，而非只看上一步。

### 1.7 MAIC: Multi-Agent Incentive Communication via Decentralized Teammate Modeling
Yuan et al., AAAI 2022 (no arXiv). https://ojs.aaai.org/index.php/AAAI/article/view/21179

**How:** Each agent learns a model of every teammate and generates messages that directly bias each teammate's Q-values (an "incentive"), rather than sharing raw observations. A sparsity regulariser prunes unnecessary messages. The method plugs into value factorisation (QMIX, QPLEX).
**Result:** Improves over QMIX and other communication baselines on SMAC, Hallway and level-based foraging. Without teammate modelling, the ablation drops back to roughly QMIX level.
**中文：** 每个智能体建模队友，并生成直接调整队友 Q 值的"激励"消息，而不是共享原始观测；在 SMAC 上优于 QMIX，去掉队友建模后性能退回 QMIX 水平。
**Relevance:** This argues for messages that are imperatives about what teammates should do ("attack e3") rather than reports of what I am doing ("I attack e3"). Our LLMs currently produce the second kind. / 支持"指令式"消息（"打 e3"）而非"汇报式"消息（"我在打 e3"），我们的 LLM 目前多为后者。

### 1.8 MASIA: Efficient Communication via Self-supervised Information Aggregation
Guan et al., 2023, extended arXiv version of the NeurIPS 2022 paper (NeurIPS proceedings). https://arxiv.org/abs/2302.09605

**How:** A permutation-invariant encoder aggregates all received messages into one compact representation. The encoder is trained self-supervised to reconstruct the global state and predict future information. Each agent then extracts the parts relevant to it. The paper also releases offline datasets for communication.
**Result:** Outperforms communication baselines in both online and offline MARL settings, including SMAC.
**中文：** 用置换不变编码器聚合收到的消息，并以自监督方式重建全局状态、预测未来；在线和离线设置下均优于基线。
**Relevance:** Supports an auxiliary "listening" objective during distillation (c): ask the 4B to predict the global state, such as enemy HP or the team's chosen target, from teammates' messages. / 在蒸馏 (c) 中加入辅助"倾听"任务：根据队友消息预测全局信息（敌方血量、团队目标）。

### 1.9 Learning Multi-Agent Communication with Contrastive Learning
Lo, Sengupta, Foerster, 2023, ICLR 2024. https://arxiv.org/abs/2307.01403

**How:** Treats messages from different agents in the same trajectory as different views of the same state. A contrastive loss maximises the mutual information between messages from the same trajectory, inducing a shared, symmetric language without a differentiable channel.
**Result:** Faster and better than prior work on communication-essential environments. Probing shows that the messages encode global state.
**中文：** 将同一轨迹中不同智能体的消息视为同一状态的不同"视角"，用对比学习最大化其互信息，诱导出共享语言。
**Relevance:** A cheap diagnostic: probe whether our units' messages from the same step "agree" (e.g. name the same target). Low agreement is a measurable coordination failure. / 低成本诊断：同一步内各单位消息是否指向同一目标，可作为可量化的协同失败指标。

### 1.10 CommFormer: Learning Multi-Agent Communication from Graph Modeling Perspective
S. Hu et al., 2024, ICLR 2024. https://arxiv.org/abs/2405.08550

**How:** Learns which agent should talk to which, as a sparse communication graph. It uses a continuous relaxation and bi-level optimisation, together with attention-based message passing, trained end to end.
**Result:** Strong results across cooperative tasks, including SMAC, and robust when the number of agents changes.
**中文：** 将"谁与谁通信"建模为可学习的稀疏图，与注意力消息传递联合端到端优化。
**Relevance:** Motivates a topology ablation: broadcast to all vs a "squad leader" whose messages everyone reads (see also [RL 4.5]). / 可做拓扑消融：全员广播 vs "队长"广播（另见 [RL 4.5]）。

## 2. Measuring whether communication matters / 如何衡量通信是否真的起作用

### 2.1 On the Pitfalls of Measuring Emergent Communication
Lowe et al., 2019, AAMAS 2019. https://arxiv.org/abs/1903.05168

**How:** The paper argues that common metrics, such as task reward with comm vs without, or speaker consistency, can be misleading. It distinguishes positive signalling (the message correlates with the speaker's state or intent) from positive listening (the message changes the listener's behaviour). It proposes causal influence of communication (CIC), measured by intervening on the message.
**Result:** Shows agents that "speak meaningfully" while listeners ignore them, and agents whose reward improves for reasons unrelated to message content.
**中文：** 区分"有效发信"（消息与说话者意图相关）和"有效倾听"（消息改变接收方行为），提出通过干预消息测量通信的因果影响（CIC）。
**Relevance:** The key design for experiment (a). Besides comm vs no-comm, add test-time interventions: shuffle messages between episodes, replace them with a wrong target, or blank them. Then measure the change in each listener's target choice. This directly tests our observation that units "announce rather than coordinate". / 实验 (a) 的关键设计：除有无通信外，在测试时打乱、篡改或清空消息，测量接收方目标选择的变化，直接检验"只宣布不协同"的观察。

### 2.2 Social Influence as Intrinsic Motivation for Multi-Agent Deep RL
Jaques et al., 2018, ICML 2019. https://arxiv.org/abs/1810.08647

**How:** Each agent receives an intrinsic reward equal to its causal influence on other agents' actions. This is computed counterfactually as the KL divergence between a teammate's action distribution given my actual action or message and its distribution marginalised over alternatives. A learned model of other agents makes this decentralised.
**Result:** Influence rewards lead to more coordination and emergent meaningful communication in sequential social dilemmas.
**中文：** 以"对他人动作的因果影响"（反事实 KL 散度）作为内在奖励，促进协同与有意义的通信。
**Relevance:** Directly computable with an LLM. Score a teammate's action log-probs with the message and with a counterfactual message (or none); the difference is a per-message influence reward for GRPO in (d). / 可直接用 LLM 计算：比较队友在真实消息与反事实消息下的动作对数概率差，作为 (d) 中逐条消息的影响力奖励。

## 3. Centralised training, decentralised execution in SMAC / SMAC 中的 CTDE

### 3.1 QMIX: Monotonic Value Function Factorisation
Rashid et al., 2018, ICML 2018. https://arxiv.org/abs/1803.11485

**How:** Each agent has its own Q-network over local observations. A mixing network, whose weights are produced from the global state by hypernetworks, combines the agent Q-values into a team Q_tot that is monotone in each of them. Training uses the team reward; execution is fully decentralised, with each agent taking its argmax.
**Result:** The standard SMAC baseline. Learns focus fire and kiting from team reward alone, with no communication at execution time.
**中文：** 各单位局部 Q 网络经由全局状态条件化的单调混合网络合成团队 Q 值；仅凭团队奖励即可学到集火与拉扯，执行时无需通信。
**Relevance:** Shows that coordination like focus fire can come from training with a shared team signal, not only from talk. For (d), the reward should be the team outcome, plus damage dealt, so that individual units are not rewarded for selfish kills. / 说明集火可由"共享团队信号的训练"产生而不必依赖通信；(d) 应使用团队奖励。

### 3.2 MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
C. Yu et al., 2021, NeurIPS 2022 Datasets & Benchmarks. https://arxiv.org/abs/2103.01955

**How:** PPO with a shared actor across agents and a centralised value function that sees global state (CTDE). The keys are implementation details: value normalisation, agent-specific global state, few epochs, small clip, and death masking.
**Result:** Matches or beats off-policy value-factorisation methods on SMAC, MPE, Hanabi and Google Research Football.
**中文：** 参数共享策略 + 看全局状态的集中价值函数；依赖若干实现细节，在 SMAC 等任务上与值分解方法相当或更好。
**Relevance:** This is exactly our shared-weights setting. If we use PPO rather than GRPO, a value head that sees the global SMACv2 state (only at training time) is the standard recipe. Death masking also matters, because dead units should produce no gradient. / 与我们的共享权重设置一致；若用 PPO，训练时价值头可看全局状态；死亡单位应被屏蔽。

### 3.3 ROMA: Multi-Agent Reinforcement Learning with Emergent Roles
T. Wang et al., 2020, ICML 2020. https://arxiv.org/abs/2003.08039

**How:** Learns a stochastic role embedding per agent, conditioned on its trajectory, and regularises roles to be identifiable and specialised. Agents with similar roles share similar policies.
**Result:** Improves on QMIX on hard SMAC maps, with interpretable role clusters.
**中文：** 为每个单位学习随轨迹变化的角色嵌入，使角色可辨识且专门化；在 SMAC 难图上优于 QMIX。
**Relevance:** With shared weights, all units reason identically, so they may pick different "weakest" enemies depending on their local view. Assigning roles in the prompt (e.g. "you are unit 2 of 5; leader = unit 1") is a cheap way to break symmetry. / 共享权重导致单位推理同质；在提示中指定角色（如"1 号为队长"）是打破对称的低成本方法。

### 3.4 MAT: Multi-Agent Reinforcement Learning is a Sequence Modeling Problem
Wen et al., 2022, NeurIPS 2022. https://arxiv.org/abs/2205.14953

**How:** Uses the multi-agent advantage decomposition theorem. Agents choose actions sequentially within a step, and agent i conditions on the actions already chosen by agents 1..i-1 through a transformer decoder. This turns the joint action into an autoregressive sequence and gives monotonic improvement guarantees.
**Result:** State of the art on SMAC and other benchmarks at the time, and strong few-shot transfer across agent counts.
**中文：** 利用多智能体优势分解定理，同一时刻内各单位依次决策，后一单位以前面单位已选动作为条件，把联合动作变成自回归序列。
**Relevance:** **The most actionable idea for us.** Instead of messages delayed by one step, let units act in a fixed order within a step, with each prompt containing "allies already chose: u1→e3, u2→e3". The coordination information then arrives the same step, and same-step focus fire becomes trivial to express. It is also an inference-time change, which makes it testable in (a) and (b) without training. / **最可操作的启示**：让单位在同一步内按顺序决策，提示中写入"队友已选：u1→e3, u2→e3"，使协调信息当步可用，无需训练即可在 (a)(b) 中测试。

## 4. Coordination with partners and conventions (Hanabi / ZSC) / 搭档协调与约定

### 4.1 Other-Play for Zero-Shot Coordination
H. Hu et al., 2020, ICML 2020. https://arxiv.org/abs/2003.02979

**How:** Self-play is trained under random relabellings of the environment's symmetries, such as card colours. This prevents agents from adopting arbitrary conventions that only their own training copy understands.
**Result:** Much better cross-play with independently trained agents and with humans in Hanabi.
**中文：** 在环境对称性随机重标记下进行自博弈，避免学到只有自身副本能懂的任意约定；在 Hanabi 中显著提升与陌生搭档和人类的协作。
**Relevance:** A warning for self-play RL (d). Shared-weight units can drift into private message codes; see the collusion work in [RL §5]. Randomly permuting enemy and ally IDs in prompts during training is the analogue of other-play. / 对 (d) 的警示：共享权重单位可能形成私有暗语；训练时随机置换敌我编号即 other-play 的对应做法。

### 4.2 InstructRL: Language Instructed Reinforcement Learning for Human-AI Coordination
H. Hu & Sadigh, 2023, ICML 2023 (PMLR 202). https://arxiv.org/abs/2304.07297

**How:** An LLM, conditioned on a natural-language instruction (e.g. "hint the most recent card"), gives a prior policy. RL is regularised toward that prior, so the learned equilibrium follows the instructed convention.
**Result:** Converges to human-aligned conventions in Hanabi and improves human-AI coordination in user studies.
**中文：** 用 LLM 根据自然语言指令生成先验策略，并以其正则化 RL，使学到的均衡符合指定约定。
**Relevance:** Supports putting the convention itself in the system prompt ("focus fire: everyone attacks the lowest-HP visible enemy; announce it"). For (d), add a KL penalty to the prompted base model so RL refines that convention instead of drifting. / 支持把约定写入系统提示（"集火：全员打最低血量敌人"），(d) 中对提示化基座模型加 KL 约束。

### 4.3 Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents
Ramesh et al., 2026, preprint. https://arxiv.org/abs/2601.18077

**How:** Benchmarks 17 LLMs in 2-5 player Hanabi under three levels of context scaffolding: minimal, programmatic Bayesian deductions, and working memory. It releases game logs and dense move-level utilities (HanabiRewards), then applies SFT and RL to a 4B model (Qwen3-Instruct).
**Result:** SFT improves cooperative play by 21% and RL with dense move-level rewards by 156%, reaching within about 3 points of o4-mini. Scaffolding with explicit deductions helps most.
**中文：** 在 Hanabi 中评测 17 个 LLM 和三种上下文脚手架；对 4B 模型做 SFT 和基于逐步效用的 RL，分别提升 21% 和 156%。
**Relevance:** The closest evidence that a **4B model can be RL-trained to cooperate**, with the gain coming from **dense per-move rewards** rather than episode outcomes. For (d), compute per-step utilities (e.g. damage to the focus target, kill credit). For the prompt, pre-compute deductions such as "lowest-HP enemy in range of most allies". / 最接近的证据：**4B 模型可通过 RL 学会合作**，且收益来自**逐步稠密奖励**；可在提示中预先计算"多数队友射程内的最低血量敌人"。

## 5. LLMs inside MARL: language messages, LLM-designed protocols, LLM credit / LLM 与 MARL 结合

### 5.1 Verco: Learning Coordinated Verbal Communication for MARL
D. Li et al., 2024, preprint. https://arxiv.org/abs/2404.17780

**How:** Each agent is an LLM with a message module and an action module, using separate LoRA adapters. A teacher model with the **global view** generates message labels, and the student's message module is trained on them by SFT. The action module reads the messages and the local observation, and is trained with RL.
**Result:** Better learning efficiency and performance than existing methods on Overcooked, plus interpretable messages.
**中文：** LLM 智能体分为消息模块与动作模块（各自 LoRA）；由看全局的教师模型生成消息标签，SFT 训练学生的消息模块，动作模块用 RL 训练。
**Relevance:** A direct recipe for (c). Have Gemma 31B, given the global state, write the ideal `say` message, and distil it into the 4B separately from actions. Separate LoRA adapters for speaking and acting are worth an ablation. / (c) 的直接配方：让 31B 看全局状态写"理想消息"，单独蒸馏给 4B；说话与行动分用 LoRA 值得消融。

### 5.2 Towards Language-Augmented Multi-Agent Deep Reinforcement Learning
Toquebiau et al., 2025, ECAI 2025. https://arxiv.org/abs/2506.05236

**How:** Agents are trained to act, and also to produce and interpret natural-language descriptions of their observations. Language serves as both the communication channel and an auxiliary representation-learning objective. Ablations separate "learns language but does not communicate" from "communicates with oracle language".
**Result:** Beats emergent-communication baselines, gives more informative representations, and generalises better to new partners.
**中文：** 智能体在行动之外还学习生成和理解对自身观测的自然语言描述，语言既是通信又是辅助表征目标；消融区分"学语言"与"用语言通信"。
**Relevance:** Its ablation design separates two effects: language as training signal and language as channel. We can copy it in (c): distil with messages but test without them, and vice versa. / 可借鉴消融设计：区分"语言作为训练信号"与"语言作为信道"，在 (c) 中带消息训练、无消息测试，反之亦然。

### 5.3 LMAC: LLM-Guided Communication for Cooperative MARL
Bae et al., 2026, ICML 2026. https://arxiv.org/abs/2605.18077

**How:** An LLM designs the communication protocol, deciding what each agent should send. It refines the protocol iteratively, using an explicit state-awareness criterion: how accurately and how uniformly all agents can reconstruct the underlying state from their observations plus messages.
**Result:** Better state reconstruction across agents and substantial gains over prior communication baselines on several MARL benchmarks.
**中文：** 由 LLM 设计通信协议，并以"各智能体能否准确且均匀地重建全局状态"为标准迭代改进。
**Relevance:** Suggests a protocol-design loop for our `say` channel. Ask 31B to propose a fixed message schema (e.g. `TARGET e# HP% | SEEN e#,e#`) and keep the schema that best lets teammates reconstruct enemy positions and HP. This is cheaper than RL. / 可让 31B 设计固定消息模板（如 `TARGET e# HP% | SEEN e#`），选择最能让队友重建敌情的模板，比 RL 便宜。

### 5.4 Speaking the Language of Teamwork: LLM-Guided Credit Assignment in MARL
M. Lin et al., 2025, preprint. https://arxiv.org/abs/2502.03723

**How:** An LLM, given a natural-language task description, ranks or scores each agent's contribution. A potential-based, agent-specific dense reward is learned from many such queries, which reduces the effect of individual ranking errors. This replaces value decomposition for sparse-reward credit assignment.
**Result:** Faster convergence and higher returns than MARL baselines.
**中文：** 让 LLM 根据任务描述评价各智能体贡献，学习基于势函数的个体稠密奖励，以替代稀疏奖励下的值分解。
**Relevance:** For (d), a 31B judge could score each unit's step ("did this unit join the focus target?") to give per-unit credit. Keep it potential-based so the optimal policy is unchanged. / (d) 中可用 31B 作为评审为每个单位每步打分（"是否加入集火"），用势函数形式保证最优策略不变。

### 5.5 SMAC-R1: The Emergence of Intelligence in Decision-Making Tasks
Deng et al., 2024, preprint. https://arxiv.org/abs/2410.16024

**How:** Instead of per-step LLM control, a large LLM writes **decision-tree code** for the whole team from a task description, and refines it with environment reward feedback (self-reflection). The resulting scripts are distilled into Qwen2.5-7B by SFT, then improved with GRPO on script generation.
**Result:** High-quality, interpretable scripts on 23 SMAC tasks and 10 new tasks, which transfer to similar maps without modification.
**中文：** 不逐步控制，而是让 LLM 为全队编写决策树代码并依据奖励反思改进，再用 SFT 和 GRPO 蒸馏到 7B 模型。
**Relevance:** A strong baseline for (b), and a contrast to our per-unit design. A team-level script hard-codes focus fire, while per-unit LLMs must rediscover it. Worth one comparison row: an "LLM-written focus-fire script". / (b) 的强基线及对照：团队级脚本可写死集火，而逐单位 LLM 需重新发现。

## 6. RL fine-tuning LLMs for cooperation and communication / 用 RL 训练 LLM 合作与通信

### 6.1 Training Language Models for Social Deduction with Multi-Agent RL [RL 4.1]
Sarkar et al., 2025, AAMAS 2025. https://arxiv.org/abs/2502.06060

**How:** Communication is split into two skills. **Listening** is trained as supervised prediction of hidden world information (who the impostor is) from the discussion. **Speaking** is trained with multi-agent RL, rewarding each message by its **influence on other agents' beliefs**. No human demonstrations are used.
**Result:** In an Among-Us-style embodied game, it substantially improves win rate over standard RL, and the discussion is interpretable.
**中文：** 将通信拆为"倾听"（从讨论中预测隐藏信息的监督任务）与"说话"（以对他人信念的影响为奖励的多智能体 RL）。
**Relevance:** The best template for training our channel in (d). Listening: predict "which enemy will the team focus?" or enemy HP from received messages. Speaking: reward messages that raise teammates' probability of choosing the focus target. / (d) 训练信道的最佳模板：倾听 = 从消息预测团队目标；说话 = 奖励能提高队友选中集火目标概率的消息。

### 6.2 MAGRPO [RL B1] and AT-GRPO / Stronger-MAS [RL B2]
S. Liu et al., 2025 (https://arxiv.org/abs/2508.04652); Zhao et al., 2025 (https://arxiv.org/abs/2510.11062)

**How:** MAGRPO frames LLM collaboration as a Dec-POMDP and trains all agents on a **shared team reward** with group-relative advantages (CTDE-style). AT-GRPO points out that standard GRPO grouping breaks when prompts differ by role and turn, and groups rollouts **per agent and per turn**. It supports both a single shared policy and separate per-role policies.
**Result:** AT-GRPO raises long-horizon planning accuracy from 14-47% (single-agent RL baselines) to 96-99.5%. Both papers find that multi-agent training is essential for collaboration.
**中文：** MAGRPO 以共享团队奖励和组相对优势训练所有智能体；AT-GRPO 按"智能体×轮次"分组计算优势，支持共享或分角色策略。
**Relevance:** The algorithmic default for (d): shared LoRA, a team reward, and advantages computed within groups of the same (unit, step) state across sampled continuations. / (d) 的默认算法：共享 LoRA + 团队奖励 + 按（单位, 步）分组计算优势。

### 6.3 Note: GRPO does not automatically produce coordination [RL 6.2]
Hasan et al., 2026, https://arxiv.org/abs/2606.07845. In the dining-philosophers setting, high reward coexisted with coordination failure, and naive GRPO fine-tuning did not close the gap.
**中文：** 高奖励可以掩盖协同失败，朴素 GRPO 不能自动带来协同。
**Relevance:** Report focus-fire rate (fraction of team damage on one target), overkill, and the time-to-kill of the first enemy alongside win rate. / 除胜率外，应报告集火率、过量伤害和首杀时间。

## 7. LLM agents coordinating through language (no or light training) / 通过语言协同的 LLM 智能体

### 7.1 CoELA: Building Cooperative Embodied Agents Modularly with LLMs
H. Zhang et al., 2023, ICLR 2024. https://arxiv.org/abs/2307.02485

**How:** A modular agent with perception, memory (including what teammates said and did), a **communication module** that decides what to say and when (communication is costly), a planning module and execution. The setting is decentralised, with partial observability, in C-WAH and TDW-MAT.
**Result:** GPT-4-driven CoELA beats strong planning baselines and shows emergent effective communication. Open models such as LLaMA-2 underperform, but fine-tuning them on CoELA data helps. Humans trust the agents that communicate.
**中文：** 模块化具身智能体（感知、记忆、通信、规划、执行），通信有成本、由模块决定何时说什么；GPT-4 版本优于规划基线，开源小模型需微调。
**Relevance:** Matches our setting (decentralised, partial observability, costly chat). The finding that small open models need fine-tuning on strong-model data supports (c). / 设置与我们一致；小模型需在强模型数据上微调，支持实验 (c)。

### 7.2 RoCo: Dialectic Multi-Robot Collaboration with LLMs
Mandi et al., 2023 (venue not verified here; arXiv preprint). https://arxiv.org/abs/2307.04738

**How:** Each robot arm is an LLM agent. The agents hold a multi-round dialogue to propose and agree on subtasks and waypoints. The agreed plan is validated by environment feedback (collision and IK checks), and any failure is fed back into the dialogue for re-planning.
**Result:** High success on the RoCoBench multi-robot tasks, and the dialogue adapts to failures.
**中文：** 每个机械臂为一个 LLM，多轮对话协商子任务与路径点，环境检查失败后反馈回对话重新规划。
**Relevance:** Suggests an inexpensive "propose → agree" phase. For example, every K steps, run a short discussion round in which units must agree on a single focus target before acting. / 可每 K 步加入一轮简短协商，要求单位就唯一集火目标达成一致后再行动。

### 7.3 Scalable Multi-Robot Collaboration with LLMs: Centralized or Decentralized Systems?
Y. Chen et al., 2023 (venue not verified here; arXiv preprint). https://arxiv.org/abs/2309.15943

**How:** Compares four LLM communication frameworks on coordination-heavy tasks with increasing numbers of agents: centralised planner, fully decentralised dialogue, and two hybrids (a central planner whose plan agents can comment on).
**Result:** A hybrid framework gives the best success and scales best with agent count, at good token efficiency.
**中文：** 比较集中、分散及两种混合的 LLM 通信框架；混合框架成功率最高、扩展性最好。
**Relevance:** Motivates a "commander" condition for (a) and (b): one 31B call proposes a team target and the units act locally. Compare it with our fully decentralised chat. / 为 (a)(b) 增加"指挥官"条件：一次 31B 调用给出团队目标，各单位本地执行，与完全分散聊天对比。

### 7.4 ProAgent: Building Proactive Cooperative Agents with LLMs
C. Zhang et al., 2023, AAAI 2024. https://arxiv.org/abs/2308.11339

**How:** The LLM explicitly infers the teammate's current intention from its observed behaviour (belief), plans its own action to complement it, then verifies and corrects the belief as the teammate keeps acting. No communication is involved; coordination comes from modelling the teammate.
**Result:** Outperforms self-play and population-based training baselines when cooperating with AI partners in Overcooked-AI.
**中文：** LLM 根据队友行为显式推断其意图，规划互补动作，并根据后续行为修正信念；无需通信。
**Relevance:** Provides a no-channel control for (a). Show each unit its allies' last attack targets, which are visible on the field. That isolates "observed actions" from "messages". / 为 (a) 提供无信道对照：在观测中给出队友上一步的攻击目标，区分"观察到的动作"与"消息"的作用。

### 7.5 LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in LLMs
Agashe et al., 2023, Findings of NAACL 2025 (ACL Anthology). https://arxiv.org/abs/2310.03903

**How:** Agentic play in four pure-coordination games (including Overcooked and Hanabi), plus CoordQA, which probes environment comprehension, theory-of-mind reasoning and joint planning separately.
**Result:** LLMs do well when coordination depends on environment state, but poorly when it requires reasoning about partners' beliefs. They are robust to unseen partners (zero-shot coordination) compared with RL agents.
**中文：** 在纯协调游戏和问答中评测；LLM 擅长依赖环境状态的协调，但在需要推理队友信念时表现差。
**Relevance:** Explains our finding. Focus fire becomes easy if it is environment-driven (everyone picks "lowest HP visible"), but hard if units must guess what teammates will pick. Make the target rule environment-computable. / 解释我们的发现：集火若由环境决定（"打最低血量"）则容易，若需猜测队友选择则难；应让目标规则可由环境计算。

### 7.6 Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with LLMs
Cross et al., 2024, ICLR 2025 (ICLR proceedings). https://arxiv.org/abs/2407.07086

**How:** A theory-of-mind module generates natural-language hypotheses about other agents' strategies. It scores the hypotheses by how well they predict the agents' observed behaviour, then refines them. The best hypothesis conditions a high-level planner.
**Result:** Better than LLM-agent and RL baselines across competitive, mixed-motive and collaborative Melting Pot substrates.
**中文：** 生成关于他人策略的自然语言假设，按预测准确度评估与修正，最佳假设用于规划。
**Relevance:** For the 31B condition in (b), a "what will my allies attack?" step, scored against their actual next actions, is a cheap ToM scaffold. / (b) 中的 31B 可增加"预测队友会打谁"步骤，并以队友实际动作校验。

### 7.7 ReAd: Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration
Y. Zhang et al., 2024, ACL 2025. https://arxiv.org/abs/2405.14314

**How:** Learns a **sequential multi-agent advantage function** by critic regression on LLM-planned data. The LLM planner is then used as an optimiser: it proposes actions, and the advantage function accepts them or prompts refinement. This extends advantage-weighted regression to multi-agent systems.
**Result:** Higher success and fewer environment steps and LLM queries on Overcooked-AI and a hard RoCoBench variant.
**中文：** 在 LLM 规划数据上回归学习序贯多智能体优势函数，让 LLM 生成能最大化该优势的动作，减少交互与查询次数。
**Relevance:** A middle path between imitation (c) and full RL (d). Train a small critic on 31B and 4B rollouts, then rerank or filter the 4B's sampled actions with it (advantage-weighted BC). / 介于 (c) 与 (d) 之间：用小型评论家对 4B 采样动作重排序或做优势加权 BC。

## 8. Do LLM messages actually matter? Ablations and failure modes / LLM 消息真的重要吗？

### 8.1 Communication Enables Cooperation in LLM Agents
Madmoun & Lahlou, 2025, EACL 2026. https://arxiv.org/abs/2510.05748

**How:** Compares a minimal **one-word cheap-talk** channel against curriculum learning across social dilemmas, including a 4-player Stag Hunt.
**Result:** One-word messages raise Stag Hunt cooperation from 0% to 96.7%, while a game curriculum *reduced* payoffs by 27.4% in an iterated public-goods game.
**中文：** 单词级"廉价对话"使四人猎鹿博弈合作率从 0% 提升到 96.7%；课程学习反而使收益下降 27.4%。
**Relevance:** Evidence that **short, focal messages** solve coordination games. Focus fire is a coordination game (any common target beats a split). A constrained message format, such as one enemy ID, may outperform our 12-word free text. / 集火本质是协调博弈；**极短、聚焦**的消息（只说一个敌人编号）可能优于 12 词自由文本。

### 8.2 Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate
Wynn, Satija, Hadfield, 2025, ICML 2025 MAS Workshop. https://arxiv.org/abs/2509.05396

**How:** Runs multi-agent debate with heterogeneous model capabilities and tracks how answers change over rounds.
**Result:** Debate can *lower* accuracy over time, even when the stronger models are the majority. Agents switch from correct to incorrect answers to agree with their peers (sycophancy and conformity).
**中文：** 多智能体辩论可能随轮次降低准确率，模型常为附和同伴而由对改错。
**Relevance:** Conformity can be useful for focus fire, since everyone follows the first caller, but harmful if the first caller picks a bad target. Measure "follows teammate even when the target is out of range or nearly full HP". / 从众对集火有利（大家跟随第一个喊目标的人），但若目标选错则有害；应测量"即使目标不可达也跟随"的频率。

### 8.3 Talk is Cheap, Communication is Hard: Dynamic Grounding Failures and Repair in Multi-Agent Negotiation
Yao, Zou, Hawkins, 2026, preprint. https://arxiv.org/abs/2605.01750

**How:** An iterated two-agent resource-allocation game with verifiable joint optima, analysing how dialogue succeeds or fails across turns.
**Result:** Individually each agent can find the Pareto-optimal plan, yet pairs consistently fail. The failure modes are lost history, anchoring on early proposals, defaulting to equal splits, and referential binding errors. The bottleneck is joint plan formation, commitment and execution.
**中文：** 个体能找到最优方案但双方协作失败；失败模式包括遗忘历史、锚定早期提议、默认平分和指代错误；瓶颈在于共同计划的形成、承诺与执行。
**Relevance:** Names our observed failure: announcing is not committing. Log "said X, did Y" consistency and referential errors (wrong enemy ID) in the unit transcripts. / 精确命名我们的现象："宣布"不等于"承诺"；应记录"说 X 做 Y"的一致性与敌人编号指代错误。

### 8.4 Emergent Coordination in Multi-Agent Language Models
Riedl, 2025, preprint. https://arxiv.org/abs/2510.05174

**How:** An information-theoretic test (partial information decomposition of time-delayed mutual information) for whether a group of LLM agents is a "higher-order collective" or just an aggregate. Applied to a guessing game with no direct communication, under prompt interventions.
**Result:** Plain agents show little cross-agent alignment. Personas plus the instruction "think about what other agents might do" produce identity-linked differentiation and goal-directed complementarity.
**中文：** 用信息分解检验 LLM 群体是否形成"高阶集体"；人格设定加"思考他人会怎么做"的提示可产生互补协同。
**Relevance:** A cheap prompt condition for (a): add "think about which enemy your allies will attack". Synergy or mutual-information measures between units' target choices give a quantitative coordination score. / (a) 的低成本提示条件："想想队友会打哪个敌人"；单位目标选择间的互信息可作协同度量。

---

## What to borrow for our experiments / 可借鉴到我们实验的做法

Ranked by expected payoff per unit of effort. / 按"投入产出比"排序。

**1. Same-step sequential intentions, not next-step chat (MAT 3.4, AT-GRPO 6.2, hybrid planning 7.3) → experiments (a), (b).**
Process the 5 units in a fixed order within each step. Each prompt includes the targets already chosen by earlier units this step ("u1→e3, u2→e3"). This turns coordination into a same-step, environment-computable decision (LLM-Coordination 7.5 shows LLMs are good at those), and it needs no training. Run a three-arm comparison: no channel / next-step `say` / same-step sequential intents. Add a fourth arm with a "commander" call that sets one team target.
**中文：** 每步内按固定顺序处理 5 个单位，提示中写入"本步前面队友已选目标"，把协调变成当步、可由环境计算的决策，无需训练。做三组对比：无信道 / 下一步 `say` / 当步顺序意图；可加"指挥官"第四组。

**2. Causal message ablations and behavioural coordination metrics (Lowe 2.1, Jaques 2.2, 6.3, 8.3) → (a), and the evaluation for all experiments.**
Beyond comm vs no-comm, intervene at test time: shuffle messages across episodes, replace the named target with a random enemy, or blank them. Measure the change in listeners' target choice (CIC), focus-fire rate, overkill, time-to-first-kill, and "said X, did Y" consistency. This turns "units announce but don't coordinate" into numbers.
**中文：** 测试时打乱、篡改或清空消息，测量接收方目标变化（CIC）、集火率、过量伤害、首杀时间和"说 X 做 Y"一致性，把"只宣布不协同"量化。

**3. Constrained, imperative message schema (TarMAC 1.3, MAIC 1.7, Madmoun 8.1, LMAC 5.3, InstructRL 4.2) → (a), (b).**
Replace 12-word free text with a schema such as `FOCUS e<id> | SEEN e<id>:<hp>`, plus a prompt-level convention ("attack the lowest-HP enemy any ally reports"). One-word cheap talk produced near-perfect Stag Hunt cooperation, and imperative messages that change teammates' values beat reports. Have 31B propose candidate schemas and keep the one that maximises focus-fire rate.
**中文：** 用结构化指令式消息（`FOCUS e<id> | SEEN e<id>:<hp>`）替代自由文本，并在提示中写明集火约定；让 31B 提出候选模板，选集火率最高者。

**4. Distil the message channel from a global-view teacher, with a listening auxiliary (Verco 5.1, MASIA 1.8, language-augmented MARL 5.2) → (c).**
When generating 31B teacher data, also give 31B the global state and have it write the ideal `say` message and target. SFT the 4B on (local obs + messages → message, action), with an auxiliary "predict the team's focus target / enemy HP from messages" loss. Ablate messages at train time vs test time to separate language-as-signal from language-as-channel.
**中文：** 生成教师数据时让 31B 看全局状态写出理想消息和目标，SFT 4B 并加入"根据消息预测团队目标/敌方血量"的辅助任务；分别在训练和测试时消融消息。

**5. Team reward plus dense coordination credit for multi-agent GRPO (MAGRPO/AT-GRPO 6.2, Sarkar 6.1, Jaques 2.2, Hanabi 4.3, LLM credit 5.4, QMIX/MAPPO 3.1-3.2) → (d).**
Use a shared LoRA, a shared team reward, and advantages grouped per (unit, step). Add two dense terms: (i) a per-step focus-fire utility (damage to the team's focus target, kill credit), since dense move-level rewards gave the 4B Hanabi agent +156% vs +21% for SFT; and (ii) a speaker-influence reward, the increase in teammates' log-prob of attacking the focus target given the message vs a blank message. Keep a KL anchor to the prompted policy, permute unit and enemy IDs (other-play 4.1) to discourage private codes, and report behavioural metrics because GRPO alone may not close the coordination gap (6.3).
**中文：** 共享 LoRA + 团队奖励 + 按（单位, 步）分组优势；加入 (i) 逐步集火效用（Hanabi 4B 中稠密奖励带来 +156%）与 (ii) 说话影响力奖励（有消息与空消息时队友攻击集火目标的对数概率差）；保留对提示策略的 KL 约束，随机置换编号以防私有暗语，并报告行为指标。
