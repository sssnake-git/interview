# Deepseek

- Q1 **什么是MoE?**
    - reference: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

- Q2 **什么是RLHF?**
    - RLHF, 全称是Reinforcement Learning from Human Feedback(从人类反馈中强化学习), 是一种用于训练大型语言模型(LLMs)的技术, 目的是让模型的输出结果更符合人类的偏好和价值观.
        - 简单来说, RLHF就是教导AI模型像人类一样思考和判断. 
        - RLHF的核心思想是
            - 收集人类反馈
                - 让人类标注者对模型生成的不同结果进行评价和排序, 指出哪些结果更好, 哪些结果更差. 
                - 给大模型一个prompt提示, 大语言模型的初始版本可能会生成各种各样的回复, 有些好, 有些不好.
                - 人类根据大语言模型的回答给出反馈:
                    - 正反馈: 给出正面反馈, 表示这个结果好.
                    - 负反馈: 给出负面反馈, 表示这个结果不好.
                - Key points: 不是直接告诉大模型某个具体动作分解(例如某个问题要怎么分解步骤), 而是通过反馈引导大模型逐渐学会某个问题的含义和期望的动作.
            - 训练奖励模型
                - 利用这些人类反馈数据, 训练一个 奖励模型, 让这个模型学会 预测 什么样的模型输出更受人类欢迎. 
                - 大模型逐渐学习, 做出某一步推理时, 会得到奖励, 知道什么推理步骤是好的或者不好的.
                - 在RLHF中, 奖励模型扮演了"大模型的理解"的角色, 通过学习大量的人类反馈数据(哪些是好的, 哪些是不好的), 模拟人类的偏好.
                - 奖励模型输出, 对于一个新的模型输出, 奖励模型可以给出一个评分(奖励值), 来预测这个输出"符合人类偏好"的程度, 评分越高, 表示越好.
            - 强化学习优化
                - 因为某些推理步骤能够带来奖励, 模型会更倾向做出特定的推理步骤, 以获得更多的奖励, 会不断的尝试和调整自己的行为, 直到训练收敛.
                - 使用强化学习算法(例如PPO, 或者更先进的DPO), 根据奖励模型的评分, 不断调整和优化 语言模型自身, 使其生成更高奖励(即更符合人类偏好)的结果. 强化学习算法会利用奖励模型给出的评分, 来指导语言模型自身的调整和优化.
                - 优化策略: 算法会鼓励模型生成奖励评分高的输出(即人类更喜欢的输出), 抑制模型生成奖励评分低的输出(即人类不喜欢的输出). 通过不断迭代优化, 语言模型会逐渐学会生成更符合人类偏好的文本.
            - 可以把RLHF理解为给AI模型请了一位 "人类老师". 这位老师不直接告诉模型"正确答案", 而是通过 反馈(好评或差评)的方式, 引导模型逐渐学会生成人类更喜欢的结果.

    - 1 RLHF (Reinforcement Learning from Human Feedback) - 从人类反馈中强化学习

        - RLHF 是一种训练语言模型的技术, 它利用人类的偏好作为奖励信号, 以使模型的行为与人类的期望对齐. 正如 Christiano 等人 (2017) 在论文 "Deep reinforcement learning from human preferences" 中提出的:
            "We present an approach for directly training agents by optimizing them to agree with human preferences. We train a reward function to predict which of two trajectory segments a human rater will prefer. This reward function is then used to train agents using reinforcement learning."

        - 目标: RLHF 的主要目标是解决传统语言模型训练方法 (如监督学习) 难以捕捉的人类偏好和价值观. Ouyang 等人 (2022) 在 InstructGPT 论文 "Training language models to follow instructions with human feedback" 中指出:
            "Supervised fine-tuning (SFT) improves on pre-trained LMs, but can still produce outputs that are unhelpful, untruthful, or toxic.  Reinforcement learning from human feedback (RLHF) has shown promise in aligning LMs with human intent."

    - 2 PPO (Proximal Policy Optimization) - 近端策略优化 在 RLHF 中的应用
        - PPO在RLHF中的角色: PPO是一种常用的强化学习算法, 被广泛应用于 RLHF 流程中, 用于 策略模型的优化.   InstructGPT 论文 (Ouyang 等人, 2022)  详细描述了使用 PPO 算法微调语言模型以遵循指令的过程:
            "We fine-tune a pretrained language model using reinforcement learning (RL) to optimize a reward model that is trained to predict which of two outputs humans would prefer. We use the Proximal Policy Optimization (PPO) algorithm."

        - PPO的优势(在RLHF中): PPO 算法因其 稳定性和相对容易实现 而被 RLHF 领域广泛采用.  Schulman 等人 (2017) 在 PPO 论文 "Proximal policy optimization algorithms" 中强调了 PPO 的优势:
            "We propose a family of policy optimization methods that perform comparably or better than state-of-the-art policy gradient methods, while being much simpler to implement and tune. We call these methods Proximal Policy Optimization (PPO) algorithms."

    - 3 DPO (Direct Preference Optimization) - 直接偏好优化
        - DPO 的提出和目标: DPO是一种 更直接的RLHF替代方法, 旨在简化RLHF流程, 并提高训练稳定性. Rafailov等人(2023)在DPO论文"Direct preference optimization: Your language model is secretly a reward model"中提出了 DPO 算法:
            We present Direct Preference Optimization (DPO), a new algorithm for fine-tuning language models using human preferences. DPO sidesteps reward modeling and directly optimizes the policy.

        - DPO与PPO的区别: DPO的核心区别在于无需显式训练奖励模型, 而是直接从人类偏好数据中学习策略. DPO 论文 (Rafailov 等人, 2023) 进一步解释了 DPO的优势:
            DPO is simpler and more stable to train than existing RLHF algorithms like PPO, and avoids the complexities of reward modeling.

        - DPO的理论基础: DPO算法基于一个重要的理论发现, 即 语言模型本身就隐含地学习了一个奖励函数. DPO 论文(Rafailov 等人, 2023)  指出:
            We show theoretically that the optimal policy under the preference model can be recovered by a simple closed-form solution, which we call Direct Preference Optimization (DPO).

    - 4 DPO 与 PPO 的联系和对比
        共同目标: DPO 和 PPO 都旨在利用人类反馈来优化语言模型, 使其输出更符合人类偏好.  它们都属于 RLHF 的范畴. 
        PPO 的间接优化 vs. DPO 的直接优化: PPO 通过 训练奖励模型 作为中间步骤, 然后 使用奖励模型引导策略优化, 是一种 间接优化 方法.  DPO 则 直接优化策略, 无需显式奖励模型, 是一种 更直接的优化 方法. 
        简化与效率: DPO 简化了 RLHF 流程, 降低了计算成本, 并且在实践中表现出 更高的训练稳定性.  在许多情况下, DPO 可以达到与 PPO 相当甚至更好的性能, 同时训练效率更高.
        适用场景: PPO 仍然是一种 通用且强大的 RL 算法, 适用于更广泛的强化学习任务.  DPO 则 更专注于语言模型的人类偏好对齐任务, 在这一特定领域表现出优势.

    - 总结:
        RLHF 是一种利用人类反馈来训练语言模型的框架. PPO 和 DPO 都是 RLHF 框架下用于策略优化的算法. PPO 是一种通用的强化学习算法, 通过训练奖励模型来间接优化策略. DPO 是一种更简洁, 更直接的 RLHF 替代方案, 它避免了奖励模型的训练, 直接从人类偏好数据中优化策略, 在训练效率和稳定性上具有优势. DPO 可以被视为是针对语言模型偏好对齐任务而设计的, 对 PPO 的一种简化和改进.

- Q3 **什么是PPO与DPO? 二者有什么区别?**
    - 模型训练中的DPO(Direct Preference Optimization, 直接偏好优化)和PPO(Proximal Policy Optimization, 近端策略优化) 都是从人类反馈中学习 (Reinforcement Learning from Human Feedback, RLHF)的重要技术, 用于提升大型语言模型(LLM)的性能, 使其输出更符合人类偏好. 虽然它们都属于 RLHF 范畴, 目标一致, 但在方法和具体实现上存在显著的区别.

    - DPO(Direct Preference Optimization) - 直接偏好优化
        - 核心思想: DPO 是一种 **更简化, 更直接** 的 RLHF 方法, 它 **直接优化策略模型**, 使其输出结果与人类偏好数据对齐, 而 **无需显式地训练奖励模型**. 
        - 训练数据: DPO 的训练数据是 **人类偏好数据对**, 例如, 对于同一个提示 (Prompt), 人类标注者会选择模型 A 的输出比模型 B 的输出更符合偏好 (例如, 更安全, 更符合指令, 更流畅等). 这种数据形式是 **(prompt, preferred_response, rejected_response)** 三元组.
        - 优化目标: DPO 的优化目标是 **直接学习一个策略**, 使得对于人类偏好的响应, 模型输出的概率更高；对于人类不偏好的响应, 模型输出的概率更低.   它通过一个 **二元交叉熵损失函数** 来实现这个目标, 这个损失函数直接基于人类的偏好数据对进行优化.
        - 优点:
            - 简化流程: DPO **避免了PPO中复杂的奖励模型训练过程**, 简化了RLHF流程, 降低了训练的复杂度和计算成本.
            - 更稳定: DPO 的训练过程通常 **更稳定**, 更不容易出现 PPO 中可能遇到的训练不稳定问题.
            - 高效: DPO 通常 **更高效**, 在相同的计算资源下, 可能更快地达到目标性能.
        - 缺点:
            - 依赖高质量偏好数据: DPO 的性能高度依赖于 **高质量的人类偏好数据**.  如果偏好数据质量不高, 或者标注不一致, 会影响 DPO 的效果. 
            - 可能损失多样性: DPO 可能会过度拟合人类偏好数据, 导致模型输出 **多样性降低**, 可能生成更保守或更符合常见偏好的结果, 但创新性可能不足. 
            - 直接优化策略, 解释性较弱: DPO 直接优化策略模型, 中间没有显式的奖励函数, 因此在 **解释模型行为** 上可能不如 PPO 直观. 

    - PPO(Proximal Policy Optimization) - 近端策略优化 (在 RLHF 中的应用)
        - RLHF流程中的PPO, 在RLHF中, PPO处于**策略模型优化的核心阶段**, 通常在**奖励模型训练完成之后**使用. RLHF的PPO训练流程通常包括以下几个步骤:
            - 1 预训练语言模型(Pre-trained LM): 首先, 需要一个预训练好的语言模型作为基础模型.
            - 2 奖励模型训练(Reward Model Training): 使用人类偏好数据(例如, 对模型输出进行排序或评分) 训练一个**奖励模型**. 奖励模型的目标是**预测模型输出的质量或人类偏好程度**. 训练数据通常是(prompt, response, reward_score)三元组, 其中reward_score由人类标注给出.
            - 3 PPO策略优化(PPO Policy Optimization): 使用PPO算法**优化预训练语言模型(策略模型)**, 使其在生成文本时, 能够**最大化奖励模型给出的奖励值**. PPO算法**会限制策略更新的幅度**, 以保证训练的稳定性.
        - PPO的核心机制:
            - Actor-Critic架构: PPO算法通常采用**Actor-Critic 架构**. Actor(策略模型)负责生成文本, Critic(价值函数) 负责评估生成文本的价值(奖励).
            - 重要性采样(Importance Sampling): PPO使用**重要性采样**来**重用旧策略的数据**进行新策略的训练, 提高了数据利用效率.
            - 近端约束(Proximal Constraint): PPO的核心是 **近端约束机制**, 通过**KL 散度惩罚**或**裁剪 (Clipping)**等方法, **限制新策略与旧策略的差异**, 防止策略更新幅度过大, 保证训练的稳定性.
        - 优点:
            - 成熟的RL算法: PPO是一种**成熟且广泛应用的强化学习算法**, 在各种任务中都表现良好. 
            - 训练稳定: PPO的近端约束机制有助于**保证训练的稳定性**, 防止策略崩溃或震荡. 
            - 可解释性: PPO训练过程中, 显式地训练了**奖励模型**, 奖励模型可以作为一种 **可解释性工具**, 帮助理解模型偏好哪些类型的输出. 
            - 更灵活: PPO框架**更灵活**, 可以方便地集成各种奖励函数和约束条件, 适应不同的 RLHF 任务. 
        - 缺点:
            - 流程复杂: RLHF + PPO的流程**相对复杂**, 需要**训练奖励模型和策略模型两个模型**, 训练流程更长, 调试难度更高. 
            - 计算成本高: PPO训练通常**计算成本较高**, 需要更多的计算资源和时间. 
            - 超参数敏感: PPO的性能对**超参数**比较敏感, 需要仔细调整超参数才能取得最佳效果. 

    - DPO和PPO的区别和联系
        | 特征 | DPO (直接偏好优化) | PPO (近端策略优化) |
        | ----- | ----- | ----- |
        | **核心思想** | 直接优化策略, 与人类偏好对齐, 无需显式奖励模型 | 使用奖励模型引导策略优化, 最大化奖励值 |
        | **奖励模型** | **无需训练奖励模型** | **需要训练奖励模型**, 奖励模型是 PPO 训练的关键组成部分 |
        | **训练数据** | 人类偏好数据对 (prompt, preferred\_response, rejected\_response) | 奖励模型训练数据 (prompt, response, reward\_score)；PPO 训练数据 (prompt, response) |
        | **优化目标** | 直接优化策略, 使偏好响应概率更高, 非偏好响应概率更低 | 优化策略, 最大化奖励模型给出的奖励值 |
        | **训练流程** | 简化, 更直接 | 复杂, 多阶段 (奖励模型训练 + PPO 策略优化) |
        | **计算成本** | 较低 | 较高 |
        | **训练稳定性** | 通常更稳定 | 可能存在训练不稳定问题, 需要仔细调整超参数 |
        | **可解释性** | 较弱, 缺乏显式奖励函数 | 较强, 奖励模型可以作为可解释性工具 |
        | **灵活性** | 相对较低 | 更灵活, 可以集成各种奖励函数和约束条件 |

    - 联系
        - 目标一致:**  DPO 和 PPO 的 **最终目标都是一致的**, 即 **提升大型语言模型的性能, 使其输出更符合人类偏好**. 
        - RLHF 范畴:**  两者都属于 **从人类反馈中学习 (RLHF)** 的范畴, 都利用人类的偏好数据来指导模型的训练. 
        - 解决相同问题:**  DPO 可以被视为 **PPO 的一种简化和改进版本**, 它们都旨在解决如何利用人类反馈来 fine-tune 语言模型, 使其更好地服务于人类用户. 

    - 总结
        - DPO 和 PPO 都是重要的 RLHF 技术, 用于提升 LLM 的性能.  **DPO 是一种更简洁, 高效且稳定的方法, 它直接优化策略模型, 避免了奖励模型的训练**, 但在某些方面可能不如 PPO 灵活和可解释.  **PPO 是一种更成熟, 灵活且可解释的 RL 算法, 但流程更复杂, 计算成本更高, 训练难度也更大**. 
        - 选择 DPO 还是 PPO, 需要根据具体的应用场景, 资源限制, 对模型性能和稳定性的要求等因素进行权衡.  在实际应用中, DPO 因其简洁高效的特点, 近年来受到越来越多的关注和应用. 

- Q4 **什么是GRPO?**
    - DeepSeek训练中使用的*GRPO(Group Relative Policy Optimization, 群组相对策略优化)**是一种专为大规模语言模型设计的强化学习算法, 由DeepSeek团队在改进PPO(Proximal Policy Optimization)的基础上提出. 以下是其核心机制与技术特点的综合分析:
    - 1 GRPO的核心创新
        GRPO通过以下关键技术解决了传统PPO在语言模型训练中的效率与稳定性问题:
        - **无需价值网络**:传统PPO依赖价值网络(Critic)估计优势函数, 而GRPO通过**群组采样与奖励归一化**直接计算相对优势, 避免了训练额外模型的资源消耗. 
        - **群组相对优势估计**:对同一输入生成多个候选答案(群组), 通过组内奖励的均值和标准差进行归一化, 得到相对优势值(如公式:\(\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}\)), 从而替代传统优势函数. 
        - **KL散度惩罚**:在目标函数中加入KL散度项, 防止策略更新偏离参考模型(如SFT微调后的模型), 确保训练稳定性. 

    - 2. **GRPO的训练流程**
        GRPO的训练分为四个主要步骤:
        1. **生成补全**:对每个问题生成多个候选答案(如组大小\(G=4\)或更高), 形成群组样本. 
        2. **计算优势**:利用奖励模型对每个答案评分, 并进行组内归一化, 得到相对优势值. 
        3. **估计KL散度**:通过近似方法计算当前策略与参考策略的分布差异, 作为正则化项. 
        4. **优化目标函数**:结合相对优势与KL散度惩罚, 通过梯度上升更新模型参数. 目标函数示例:
        \[
        \mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^G \left[ \text{优势项} - \beta \cdot \text{KL散度} \right]
        \]
        其中, 裁剪机制(Clip)限制策略更新幅度, 防止突变. 

    - 3. **GRPO的优势与局限性** 
        **优势**:
        - **高效性**:省去价值网络, 显著降低内存与计算开销, 适合大规模模型(如DeepSeek-R1). 
        - **稳定性**:群组对比与KL惩罚机制减少了策略崩溃风险. 
        - **适用性**:在数学推理(如DeepSeekMath)等复杂任务中表现优异. 

        **局限性**:
        - **依赖参考策略**:需依赖高质量的参考模型(如SFT模型), 否则可能影响训练效果. 
        - **超参数敏感**:组大小\(G\), 裁剪阈值\(\epsilon\)等参数需精细调整. 
        - **理论分析不足**:相比PPO, 其收敛性和泛化性仍需进一步研究. 

    - 4. **实际应用案例**
        - **DeepSeekMath**:通过GRPO在数学推理任务中超越GPT-4等模型, MATH基准测试准确率达51.7%. 
        - **DeepSeek-R1**:结合GRPO与双重奖励系统(格式奖励+准确性奖励), 实现低成本高效训练, 推理成本仅为同类模型的1/30. 

    - 5. **与其他算法的对比**
        | **特性** | **PPO** | **GRPO** |
        |-----|-----|-----|
        | **价值网络需求** | 需要独立的价值网络 | 无需价值网络 |
        | **优势估计方法** | 基于GAE(广义优势估计)| 群组内奖励归一化 |
        | **内存占用** | 高(需存储价值网络参数)| 低(仅策略网络) |
        | **适用场景**  | 通用强化学习任务 | 语言模型微调, 复杂推理任务 |

    ---

    - 总结
    GRPO通过群组采样和相对优势估计的创新设计, 解决了传统强化学习算法在语言模型训练中的资源瓶颈问题. 其核心思想是**以组内对比替代价值网络**, 结合KL正则化确保稳定性, 为大规模模型的高效训练提供了新思路. 然而, 其对参考策略的依赖和参数敏感性仍需在后续研究中进一步优化.


- Q5 **Ascend部署Deepseek**
    - 算力部署情况
        - 一个Deepseek 671B"满血"模型部署需要4台服务器, 每台服务器8卡910B, 共32卡. 32卡理论显存$32 \times 64G = 2.048T$
        - Deepseek部署的是fp16(2 byte)精度的, 因此参数占用显存: $ 671\times \frac{ 10^9 \times 2}{1024^3} = 671 \times 1.865G \approx 671 \times 2G \approx 1.4T$
        - 再加上其他的KV cache缓存.

- Q6 **什么是NSA?**
    - 这篇论文介绍了一种新的方法, 称为原生稀疏注意力(NSA), 它使人工智能(AI)模型能够比传统方法更快, 更高效地处理非常长的文本. 为了完全理解这些概念, 我们将其分解为简单的部分. 对于Transformer架构, 最核心的self-attention(full-attention)需要一次性处理很长的token序列, 所消耗的算力和时间成本都会非常大, 计算复杂度是$O(N^2)$. 事实上, 人类读书的时候会自动跳读, 抓重点. 目前大模型的解决方案其实都是在打补丁, 滑动窗口法是只看当前段落的文字, 容易漏掉全局信息; 随机抽样法可能会错过关键信息; 事后压缩法是先读一遍再删减, 本质上还是浪费了第一遍的算力. 一个比较经济的办法是用稀疏注意力机制(Sparse Attention), 其sparse是相对于原来的full attention而言的, 在full attention中每个token都要跟所有的tokens进行计算, 而sparse attention时只选择部分重要的token进行计算. NSA中的native强调的是可训练的sparse attention, 设计了算法和硬件结合的技术进行实现.
    - 为什么AI需要注意力?
        - AI模型, 尤其是像ChatGPT这样的大型语言模型(LLM), 使用一种称为"注意力(Attention)"的技术来处理文本. 想象一下你在读一本书. 要理解一个句子, 你不仅要看当前的词, 还要回忆起前面句子中的相关词, 以便理解整个内容. AI也做类似的事情, 通过注意力机制帮助它确定哪些词是重要的以及它们之间的关系. 问题在于, 传统的注意力(全注意力)会查看文本中的每个词, 并将它与其他所有词进行比较. 这在处理短文本时是可以接受的, 但当文本非常长(如整本书或长法律文件)时, 这个过程变得太慢且计算成本太高. 
    - 什么是稀疏注意力? 
        - 稀疏注意力不是平等地看待每个词, 而是通过只关注最重要的词来提高效率. 可以把它想象成阅读摘要而不是整本书. 
        -  NSA的三个关键技巧: 为了很好地实现这一点, 原生稀疏注意力(NSA)引入了一种新的方式来过滤掉不重要的词, 同时保留足够的上下文以理解全文的意思. 它通过以下三种主要技术实现: 
            - 压缩(压缩粗粒度的tokens): NSA不是查看每一个单独的词, 而是将词分组为"块", 并为每个块创建一个摘要. 可以将其视为将一段话变成简短的总结. 
            - 选择(选择性保留细粒度的token): 模型从文本中挑选出应获得最多关注的最重要词. 就像学习时, 你可能会在教科书中仅突出显示关键句子一样. 
            - 滑动窗口(处理局部上下文信息): 即使NSA进行了摘要和选择, 它仍然会查看附近的词, 以确保不会错过小但重要的细节. 想象一下读书——你不会只从一页跳到另一页而不浏览附近的句子. 论文认为, 这种三步策略使NSA比传统方法快得多, 同时还能同样好(甚至更好)地理解意思.
        - 在硬件层面, NSA实现了专用内核以最大化推理的实际效率(NSA的kernel是用Triton写的):
            - 可并行的blockwise矩阵乘法
            - 优化data fetch
            - 优化数据的loop顺序
    - NSA如何工作?
        - 步骤1: 压缩(总结词组)
            - NSA不是存储每一个单独的词, 而是首先将词组压缩成概括性的"块". 想象一下你在总结书的一章. 你不会记住每一个词, 而是写几个要点来捕捉关键思想. NSA也是如此——它将词组转换为更小, 更紧凑的表示形式. 
        - 步骤2: 选择(挑选重要词)
            - 一旦NSA压缩了文本, 它会选择最相关的词进行深入处理. 想象一下在文章中突出显示最重要的句子——NSA做了类似的事情. 它不是保留每一个细节, 而是优先考虑最有意义的词. 
        - 步骤3: 滑动窗口(保持局部上下文)
            - NSA仍需要跟踪彼此靠近的词, 以防它们提供额外的意义. 想象一下阅读一个复杂的句子——你不仅仅读主要的词；还会瞥一眼前后的内容以获取完整的上下文. NSA通过在文本上滑动一个小窗口来确保捕获重要的附近信息. 
    - 为什么NSA更快?
        - NSA比传统的注意力机制快得多, 因为: 它通过只关注最重要的词减少了词之间的比较次数. 它有效地组织数据, 使计算机可以快速处理. (就像在计算机上整齐地组织文件可以更快地找到东西一样. )它针对现代计算机硬件进行了优化, 使得GPU(为AI模型提供动力)可以高效处理. 

