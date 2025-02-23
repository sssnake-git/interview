# Agent

- Q1 **什么是`Agent`?**
    - `Definition`: `Agent`(智能体)是一种不需要持续人类干预的AI代理类型或软件系统集合. 基于环境和背景信息, 自主AI代理可以解决各种问题, 做出逻辑决策, 并在没有持续人类输入的情况下处理多种任务. 
        - 特点
            -  自主`AI Agent`是根据给定的目标进行训练工作的.
            -  拥有`LLM`(大语言模型)之外的规划, 内存, 工具使用, 反思能力.
            -  具有多模态感知的能力(文本, 视频, 图像, 声音等).
            即: `Agent` = 大语言模型`+`记忆`+`规划`+`工具使用.
      
        - 简单类比的话, 可以这样子：
            - `LLM`(如`GPT`)->大脑中的某个组织(一堆神经元), 具有常识, 推理等能力.
            - 向量数据库->人脑中的感觉记忆, 短期记忆和长期记忆(`LLM`能接受的上下文非常有限, 需要外部存储辅助).
            - `Agent`->独立的人, 拥有多种感官, 可以操作外部的工具.
            - `Agent`系统->一群相互协作的智能体巧妙配合形成的团体.
    - 俗话说的好, 三个臭皮匠顶个诸葛亮, 工作中少不了集体开会讨论. 一个优秀的`Agent`系统将会有着广泛的应用前景, 这需要我们不断地进行探索和创新. 未来, 壁垒就存在于`Agent`的架构(感知, 推导, 执行等)和专属领域的数据(比如`AI`游戏里的`NPC`, 需要很长的自我经历积累, 才能形成独特的人格魅力).

- Q2 **Agent有什么应用?**
    `Autonomous AI Agent Use Cases`
    | Agent应用场景 | 解释 |
    |-----|-----|
    | 个人助理 | 完成各种任务, 如查找和回答问题, 预订旅行和其他活动, 管理日历和财务, 监控健康和健身活动. 如`WebGPT`. |
    | 软件开发 | 支持应用程序开发的编码, 测试和调试工作, 擅长自然语言输入处理任务. |
    | 交互式游戏 | 处理游戏任务, 例如创建智能的NPC, 开发自适应的反派角色和载具驾驶, 以及向玩家提供情境性指导和帮助. |
    | 预测性分析 | 实时数据分析和预测模型, 解释数据洞察, 识别模式和异常, 调整预测模型以适应不同的用例和需求. |
    | 自动驾驶 | 为自动驾驶汽车提供环境理解和图像, 提供决策指导, 支持车辆控制. |
    | 智能城市 | 技术基础, 无需人类持续维护, 特别是交通管理. |
    | 智慧客服 | 处理客户支持查询, 回答问题, 协助解决有关之前交易或付款的问题. |
    | 金融管理 | 提供研究的金融建议, 组合管理, 风险评估和欺诈检测, 合规管理和报告, 用于信用, 承保, 支付和预算管理支持. |
    | 任务生成和管理  | 生成高效的任务并执行. |
    | 智能文档处理  | 包括分类, 深度信息分析和提取, 摘要, 情感分析, 翻译和格式控制等. 例如chatPDF, chatPaper. |
    | 科学探索 | 例如, 当要求"开发一种新的抗癌药物"时, 模型提出以下处理步骤: 1.了解当前的抗癌药物发现的趋势, 2.找关键方向, 3.推荐药物目标, 4.合成化合物策略. |

- Q3 **Agent由什么组成?**
    - 引用: https://lilianweng.github.io/posts/2023-06-23-agent/
    - 一个基于LLM的自主Agent系统的几个关键组成部分
        - `Planning`(规划)
          子目标和分解: 代理可以将大型任务分解为更小, 更易管理的子目标, 从而有效地处理复杂任务.
          反思和改进：代理可以对过去的行为进行自我批评和自我反思, 从错误中学习并改进未来的步骤, 从而提高最终结果的质量. 
        - `Memory`(记忆)
          `Short-term memory`: 在prompt中或者对话上下文中的信息.
          `Long-term memory`: 这为代理提供了在长时间内保留和回忆 (无限) 信息的能力, 通常是通过利用外部向量存储和快速检索.  (比如`chatPDF`、联网搜索等) 
        - `Tool use`(工具使用)
          通过输出指令, 来调用额外的`API`, 弥补 `LLM`确实的信息和能力. 当前信息、代码执行能力、对专有信息源的访问等. 

            ![Untitled](Agent/Agent1.png)
            Overview of a LLM-powered autonomous agent system.

- Q4 **Component One: Planning**
    A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.

    - 任务拆解(Task Decomposition)
        - 思维链(Chain of Thought)
            - Think step by step, 把大任务转化为多个可管理的任务, 并阐明解释模型的思维过程, 用prompt增强大模型的复杂任务处理能力.
            - has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to "think step by step" to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
            - Inference: https://arxiv.org/pdf/2201.11903

        - 思维树(Tree of Thoughts)
            - 在推理的每个步骤都分成多步思考. 首先把复杂问题分成多个步骤, 在每一步骤再进行分解, 生成树形任务链. 在节点通过BFS和DFS进行搜索, 在节点通过分类器(prompt)或投票机制进行评估.
            - Extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
            - Inference: https://arxiv.org/pdf/2305.10601

        - 任务拆解可以这样做:
            - 1.利用简单的prompt, 例如: "Steps for XYZ...", "What are the subgoals for achieving XYZ?"
            - 2.利用特别的prompt, 例如: 如果想写小说, "Write a story line"
            - Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.

        - LLM+P
            - 依赖外部经典规划器来进行长期规划(结合领域规则), 规划步骤交给外部工具处理. LLM将问题翻译成"Problem PDDL"(Planning Domain Definition Language), 然后请求经典规划器基于现有的"Domain PDDL"生成PDDL计划, 最后将PDDL计划翻译回自然语言.
            - Another quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.

        ![Untitled](Agent/Agent2.png)
        Examples of reasoning trajectories for knowledge-intensive tasks and decision-making tasks.

    - 自我反省(Self Reflection)
        - 自我反思也是Agent的一个重要部分, 它允许Agent通过完善过去的行动决策和纠正以前的错误来迭代改进. 当出现了低效, 虚假, 长期失败的任务时, Agent可以进行停止, 优化和重置.
        - 引用: https://arxiv.org/pdf/2210.03629
        - ReAct将推理(Reasoning)和行动(Action)整合到LLM中, 这个整合的动作是通过将操作空间扩展为任务特定的离散操作和语言空间的组合来完成的. 前者使得 LLM 能与环境交互(例如使用 Wikipedia 搜索 API), 而后者则促使 LLM 以自然语言生成推理过程.
            - Thought: LLM对用户question的思考, 文字描述, 遇到了什么问题, 应该执行什么.
            - Action: 要采取的外部工具.
            - Action Input: 函数调用所需要的输入, 如执行一个Google Search所需的json参数.
            - Observation: Action执行的结果. 因此, 让GPT重复补充完成这几个部分: [Thought, Action, Action Input, Observation]就可以自主的完成复杂的任务. 也可以理解为, 让GPT增加了运算量, 用多个迭代, 一步一步的推理完成大任务.
        - 一个agent的经典prompt写法:
            ```
            ----- 提示词 -----
            "Answer the following questions as best you can. You have access to the following tools:

            Search: useful for when you need to answer questions about current events
            Music Search: A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [Search, Music Search, google search]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: what is the most famous song of christmas?
            Thought:"
            ----- 执行步骤-----
            之后,  GPT会从Thought 开始输出, 并重复输出Thought-> Action->Observation->Thought, 最终得到必备信息后, 输出最终结果. 如：
            - 第一个输出Thought：
            I should search for a popular christmas song\nAction: Music Search
            - 然后生成函数调用的入参并调用外部音乐搜索工具：
            Action: Music Search
            Action Input: most famous christmas song (搜索关键词：出名的圣诞歌) 
            - 搜索的结果, 总结后记录到Observation
            Observation: 'All I Want For Christmas Is You' by Mariah Carey.
            - 继续输出下一轮Thought:"
            最终的返回, 是得到了歌曲名：
            " I now know the final answer\nFinal Answer: 'All I Want For Christmas Is You' by Mariah Carey."
            ```

    - Reflexion
        - 引用: https://arxiv.org/pdf/2303.11366
        - 一个为AI Agents提供动态记忆和自我反思能力, 以提高推理能力的框架. 该框架采用标准的强化学习设置, 其中有一个RL反馈的过程. 在 Reflexion 框架下, 可以通过语言反馈而非更新权重的方式来强化 Language Agents. 自我反思 (Self-reflection) 是通过向 LLM 展示一个 two-shot 的示例来创建的, 其中每个示例都是成对出现的.
        ![Untitled](Agent/Agent3.png)
        Illustration of the Reflexion framework.

        - 在 Reflexion 中, 有一个 heuristic function, 用来判断 agent 推理的 trajectory 在什么时候效率降低或者出现模型幻觉, 并判断当前的 Agent 过程是否需要停止. 这里的低效 trajectory 指的是耗时过长但是又没有成功的 trajectory, 而幻觉被定义为一系列连续相同的动作导致环境中产生相同观察结果的情形.
        - Self Reflection 通过向 LLM 展示两组例子实现, 每组示例包含一对(失败轨迹 and 指导未来规划调整的 ideal Reflection). 随后, 这些 Reflection 被添加到Agent的工作记忆中(最多保留三条), 作为查询 LLM 时的上下文使用.
        ![Untitled](Agent/Agent4.png)
        Experiments on AlfWorld Env and HotpotQA. Hallucination is a more common failure than inefficient planning in AlfWorld. 

- Q5 **Component Two: Memory**

    - 首先, `Memory`从人类的记忆开始说起, 人类的记忆包含获取, 存储, 保留和后续检索信息等步骤. 在人脑中, 有几种类型的记忆:
        - 感觉记忆: 这是记忆的最早阶段, 能够在原始刺激结束后保持感觉信息(视觉, 听觉等)的印象, 感觉记忆通常只持续几秒钟, 它包括图像记忆(视觉), 回响记忆(听觉)和触觉记忆(触觉)等子类. 
        - 短期记忆(STM)或工作记忆: 它存储我们当前意识到并且需要执行复杂认知任务的信息, 比如学习和推理. 短期记忆的容量约为"7步", 持续时间为20-30秒.
            - 引用: https://labs.la.utexas.edu/gilden/files/2016/04/MagicNumberSeven-Miller1956.pdf
        - 长期记忆(LTM): 类似海马体, 长期记忆可以将信息存储很长时间, 从几天到几十年不等, 并且具有基本无限的存储容量. 长期记忆有两个子类型:
            - 显性/陈述性记忆: 这是事实和事件的记忆, 指的是那些可以有意识地回忆起来的记忆, 包括情景记忆(事件和经历)和语义记忆(事实和概念)
            - 隐性/程序性记忆: 这种记忆是无意识的, 涉及到自动执行的技能和例行程序, 比如骑自行车或打字.
        ![Untitled](Agent/Agent5.png)
        Categorization of human memory.

    - 因此, 我们将人类的记忆类型映射到计算机领域, 就可以得到下图的表格:
        | 记忆类型 | 映射 | 例子 |
        |-----|-----|-----|
        | 感觉记忆 | 学习原始输入的嵌入表示, 包括文本、图像或其他形式, 短暂保留感觉印象. | 看一张图片, 然后在图片消失后能够在脑海中回想起它的视觉印象. |
        | 短期记忆 | 上下文学习 (In Context Learning) (比如直接写入 prompt 中的信息), 处理复杂任务的临时存储空间, 受Transformer有限的上下文窗口长度限制. | 在进行心算时记住几个数字, 但短期记忆是有限的, 只能暂时保持几个项目. |
        | 长期记忆 | 在查询时智能 Agent 可以关注的外部向量存储, 具有快速检索和基本无限的存储容量. | 学会骑自行车后, 多年后再次骑起来时仍能掌握这项技能, 这要归功于长期记忆的持久存储. |
        - 目前`Agent`主要是利用外部的长期记忆, 来完成很多的复杂任务, 比如阅读`PDF`, 联网搜索实时新闻等. 标准做法是将信息的嵌入表示保存到向量数据库中.

    - 最大内积搜索(`Maximum Inner Product Search`, `MIPS`)
        - 利用外部存储可以缓解有限注意力范围的限制. 一种常见的做法是将embedding表示存储到支持快速最大内积搜索(`MIPS`)的向量数据库中. 为了优化检索速度, 通常选择近似最近邻(`Approximate Nearest Neighbors`, `ANN`)算法, 以返回近似的前`k`个最近邻, 通过牺牲少量准确性来换取巨大的速度提升.

        - 几种常用的快速`MIPS`(最大内积搜索) 算法
            - `LSH`(`Locality-Sensitive Hashing`, 局部敏感哈希)
            它引入了一种哈希函数, 使得相似的输入项以较高的概率被映射到相同的桶中, 而桶的数量远小于输入项的数量. 这种方法通过降低搜索空间来加速相似性搜索.
            - `ANN`(`Approximate Nearest Neighbors`, 近似最近邻) 
            核心数据结构是随机投影树, 这是一组二叉树, 其中每个非叶节点表示一个将输入空间分成两半的超平面, 每个叶节点存储一个数据点. 树是独立且随机构建的, 在某种程度上模仿了哈希函数. ANN 的搜索通过所有树进行, 逐步在最接近查询的一半中搜索并聚合结果. 这种方法与 KD 树的思想类似, 但更具可扩展性. 
            - `HNSW` (`Hierarchical Navigable Small World`, 分层可导航小世界) 
            该方法受“小世界网络”概念启发, 即大多数节点可以在很少的步数内被其他节点到达, 例如社交网络中的“六度分隔”特性. HNSW 构建了分层的小世界图, 其中底层包含实际数据点, 中间层提供快捷通道以加速搜索. 搜索过程从顶层的一个随机节点开始, 逐步向目标靠近. 当无法再靠近时, 移动到下一层, 直到到达底层. 上层的移动能够覆盖数据空间中的较大距离, 而底层的移动进一步优化搜索质量. 
            - `FAISS` (`Facebook AI Similarity Search`, `Facebook AI`相似性搜索)
            FAISS 假设在高维空间中, 节点之间的距离服从正态分布, 因此数据点存在聚类现象. 它通过向量量化将向量空间划分为多个簇, 并在簇内进一步精细化量化. 搜索首先通过粗量化确定候选簇, 然后在每个簇内通过精量化进一步搜索. 这种方式结合了分区和逐步精炼的过程来提高搜索效率. 
            - `ScaNN` (`Scalable Nearest Neighbors`, 可扩展最近邻) 
            `ScaNN`的主要创新点是各向异性向量量化(`anisotropic vector quantization`). 它将数据点 \( x_i \) 量化为 \( \tilde{x}_i \), 使得内积 \( \langle q, x_i \rangle \) 尽可能接近原始距离 \( \angle q, \tilde{x}_i \), 而不是简单地选择最近的量化质心点.
        
        ![Untitled](Agent/Agent6.png)
        Comparison of MIPS algorithms, measured in recall@10.


    - Some extension
        衡量最大内积搜索 (MIPS) 算法性能的指标 recall@10.
        在查询的前 10 个返回结果中, 有多少是实际的正确近邻(ground truth nearest neighbors).
        数学上, recall@10 定义为:
            \[ \text{recall@10} = \frac{\text{检索出的前 10 个最近邻中实际正确的个数}}{\text{所有正确的最近邻个数}} \]

        在 MIPS 任务中, 我们希望找到与查询向量 $q$ 具有最大内积的向量. 由于近似最近邻(ANN)方法通常会进行降维或量化, 可能导致一定的误差, $recall@10$ 用于衡量算法在前 10 个返回结果中是否能找到正确的高内积匹配项.

        - 如果一个查询的真实最近邻有 10 个, 而某个算法的前 10 个结果中有 8 个是真正的最近邻, 则:
            \[ \text{recall@10} = \frac{8}{10} = 0.8 \quad (80\%) \]
        - $recall@10$ 越高, 说明算法的召回能力越强, 搜索质量越高.

- Q6 **Component Three: Tool Use**
    The system comprises of 4 stages:

    - 任务规划, Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.
        - Instruction:
        ```
        The AI assistant can parse user input to several tasks: [{"task": task, "id", task_id, "dep": dependency_task_ids, "args": {"text": text, "image": URL, "audio": URL, "video": URL}}]. The "dep" field denotes the id of the previous task which generates a new resource that the current task relies on. A special tag "-task_id" refers to the generated text image, audio and video in the dependency task with id as task_id. The task MUST be selected from the following options: {{ Available Task List }}. There is a logical relationship between tasks, please note their order. If the user input can't be parsed, you need to reply empty JSON. Here are several cases for your reference: {{ Demonstrations }}. The chat history is recorded as {{ Chat History }}. From this chat history, you can find the path of the user-mentioned resources for your task planning.
        ```

    - 模型选取, Model selection: LLM distributes the tasks to expert models, where the request is framed as a multiple-choice question. LLM is presented with a list of models to choose from. Due to the limited context length, task type based filtration is needed.
        - Instruction:
        ```
        Given the user request and the call command, the AI assistant helps the user to select a suitable model from a list of models to process the user request. The AI assistant merely outputs the model id of the most appropriate model. The output must be in a strict JSON format: "id": "id", "reason": "your detail reason for the choice". We have a list of models for you to choose from {{ Candidate Models }}. Please select one model from the list.
        ```

    - 任务执行, Task execution: Expert models execute on the specific tasks and log results.
        - Instruction:
        ```
        With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.
        ```

    - 生成回答, Response generation: LLM receives the execution results and provides summarized results to users.
        - To put HuggingGPT into real world usage, a couple challenges need to solve: (1) Efficiency improvement is needed as both LLM inference rounds and interactions with other models slow down the process; (2) It relies on a long context window to communicate over complicated task content; (3) Stability improvement of LLM outputs and external model services.





