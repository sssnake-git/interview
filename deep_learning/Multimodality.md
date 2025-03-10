# 多模态

Q1 **多模态是什么**

- 多模态(`Multi model`)是指处理和理解来自多种模态信息. 多模态机器学习, 英文全称`Multi Modal Machine Learning`(`MMML`), 通过机器学习的方法实现处理和理解多源模态信息的能力. 目前比较热门的研究方向是图像, 视频, 音频, 文字之间的多模态学习. 多模态的目标是过分析和建模多种模式的互补性与相关性, 提升模型的理解能力和任务性能. 目前的研究热点集中在图像, 视频, 音频和语义文本等模态之间的多模态学习.

- 多模态学习的核心挑战
  - 模态表示与总结: 如何表示多模态数据, 使其能够有效捕捉各模态的独特特性和交互信息. 例如, 文本可以用词嵌入表示, 图像可以用卷积特征表示.
  - 模态之间的关联建模: 如何建立多模态之间的直接关系. 例如, 视频中的动作可以与文字描述(字幕或说明)相对应.
  - 跨模态知识迁移: 从一种模态中学习到的知识(如图像模态中的视觉特征)如何帮助另一模态的学习(如文本模态). 特别在某些模态资源不足(如缺乏标注数据)的情况下, 这种跨模态迁移尤为重要.
  - 模态间的对齐: 如何在不同模态之间实现数据对齐, 例如音频的时间序列与视频的帧序列对应.
  - 模态的异质性处理: 不同模态的数据格式, 尺度和噪声可能存在显著差异, 如何有效融合这些异质数据.

Q2 **多模态的特征融合的方式有哪些?**

- 基于简单操作的, 基于注意力的, 基于双线性池化的方法: 
    - 简单操作融合办法, 对模态间的特征进行拼接或加权求和, 后续的网络层会自动对这种操作进行自适应.
    - 基于注意力机制的融合办法, 如SAN, 双注意力网络DAN
    - 基于双线性池化的方法: 融合视觉特征向量和文本特征向量来获得一个联合表征空间, 方法是计算他们俩的外积, 这种办法可以利用这俩向量元素的所有的交互作用.
- 多模态融合的论文集: https://zhuanlan.zhihu.com/p/669017569

Q3 **多模态的预训练方式有哪些?**

- 预训练模型通过在大规模无标注数据上进行预训练, 一方面可以将从无标注数据上更加通用的知识迁移到目标任务上, 进而提升任务性能, 预训练模型在多模态任务上的运用主要有: 
    - `MLM`(`Masked Language Modeling`): 将一部分的词语进行`mask`, 任务是根据上下文推断该单词.
    - `MOC`(`Masked Object Classification`): 对图像的一部分内容进行`mask`, 任务是对图像进行分类, 此处的分类使用的依然是目标检测技术, 只是单纯的将目标检测中置信度最高的一项作为分类类别.
    - `VLM`(`Visual linguistic Matching`): 利用`[CLS]`的最终隐藏状态来预测语言句子是否与视觉内容语义匹配.
    - 图片-文本对齐(`Cross-Modality Matching`): 通过50%的概率替换图片对应的文本描述, 使模型判断图片和文本描述是否是一致的.

- 引用: [https://zhuanlan.zhihu.com/p/412126626](https://zhuanlan.zhihu.com/p/412126626)

Q4 **了解`clip`模型吗**

- `clip`模型利用`text`信息监督视觉任务自训练, 将分类任务化成了图文匹配任务, 效果可与全监督方法相当, 
- `clip`证明了简单的预训练任务, 即预测哪个标题caption与哪个图像相匹配, 是一种有效的, 可扩展的方法, 可以在从互联网上收集的4亿个(图像, 文本)对数据集上从头开始学习`SOTA`的图像表示.
- 在预训练后, 使用自然语言来引用学习到的视觉概念(或描述新的概念), 使模型`zero-shot`转移到下游任务.

- 引用: [https://towardsdatascience.com/clip-the-most-influential-ai-model-from-openai-and-how-to-use-it-f8ee408958b1](https://towardsdatascience.com/clip-the-most-influential-ai-model-from-openai-and-how-to-use-it-f8ee408958b1)

Q5 **简述什么是协同学习**

- 协同学习是指使用一个资源丰富的模态信息来辅助另一个资源相对贫瘠的模态进行学习, 协同学习是与需要解决的任务无关的, 因此它可以用于辅助多模态映射, 融合及对齐等问题的研究.
- 协同学习比较常探讨的方面目前集中在领域适应性(`Domain Adaptation`)问题上, 即如何将train domain上学习到的模型应用到 application domain.

Q6 **多模态对齐技术有哪些?**

- 模态对齐是多模态融合关键技术之一, 是指从两个或多个模态中查找实例子组件之间的对应关系, 多模态对齐方法分为显式对齐和隐式对齐两种类型. 显式对齐关注模态之间子组件的对齐问题, 而隐式对齐则是在深度学习模型训练期间对数据进行潜在的对齐.

- 显式对齐
    - 无监督方法, 如`DTW`(`Dynamic Time Warping`), 无需监督标签, 通过最小化两个序列之间的距离来实现对齐, 适用于时间序列, 多模态信号等任务. 常用于视频与音频, 语音与文本之间的对齐.
    - 监督方法: 从无监督的序列对齐技术中得到启发, 并通过增强模型的监督信息来获得更好的性能.
- 隐式对齐
    - 图像模型方法: 需要大量训练数据或人类专业知识来手动参与.
    - 神经网络方法: 在模型训练期间引入对齐机制, 通常会考虑注意力机制.
