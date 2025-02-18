# Transformer

- Q1 **Transformer为何使用多头注意力机制?**

    - Transformer架构是一种基于自注意力机制(Self-Attention)的神经网络模型, 广泛应用于自然语言处理(NLP)和计算机视觉(CV)等领域. 其核心思想是利用注意力机制来建模输入序列中的全局依赖关系, 而不依赖于传统的循环神经网络(RNN)或卷积神经网络(CNN).

    - 1 Transformer的整体架构
        - Transformer由编码器(Encoder)和解码器(Decoder)两部分组成, 适用于序列到序列(seq2seq)任务, 如机器翻译.   
        其中:
            - **编码器(Encoder)** 负责提取输入序列的语义信息, 输出高维表示.
            - **解码器(Decoder)** 负责根据编码器的输出生成目标序列.

        在"Attention Is All You Need"中:
            - 编码器和解码器均由多个相同的层堆叠而成, 每层包括多头自注意力(Multi-Head Self-Attention)和前馈神经网络(Feed Forward Network, FFN).
            - 解码器比编码器多一个掩码(Masked Multi-Head Attention)来防止看到未来的词.

    - 2 关键组成部分
        - 1 输入表示
            - Transformer的输入首先经过**词嵌入(Embedding)**, 然后加入**位置编码(Positional Encoding)**, 因为Transformer不具备RNN的序列信息, 因此需要显式加入位置信息. 

        - 2 自注意力机制(Self-Attention)
            - 自注意力的核心思想是计算输入序列中每个单词(Token)对其他单词的重要性.   
            - 在**自注意力计算**中, 每个输入token都会计算三个向量: 
                - **查询向量(Query, Q)**
                - **键向量(Key, K)**
                - **值向量(Value, V)**
            - 公式如下: 
            \[ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V \]
            - 含义:
                - \( QK^T \) 计算的是相似度(注意力分数). 
                - \( \frac{1}{\sqrt{d_k}} \) 用于防止梯度消失或爆炸. 
                - Softmax 归一化后得到注意力权重. 
                - 乘以 \( V \) 得到加权的输出. 

        - 3 多头注意力(Multi-Head Attention)
            - 在单头注意力中, 每个词只学习一种注意力模式, 而**多头注意力**(MHA)允许模型关注多个不同的语义关系. 其实现方式: 
                - 先对 \( Q, K, V \) 进行多个线性变换, 分成多个注意力头. 
                - 每个头独立执行自注意力计算. 
                - 最后将所有头的结果拼接, 并再经过一个线性变换. 

            - 公式如下: 
                \[ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O \]

        - 4 前馈神经网络(Feed Forward Network, FFN)
            - 每个编码器和解码器层中都包含一个**前馈神经网络**, 其作用是对每个词的向量表示进行非线性变换: 
                \[ \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 \]
            - 这里的 \( \max(0, \cdot) \) 是ReLU激活函数. 

        - 5 层归一化(Layer Normalization)
            - 为了稳定训练, Transformer在**多头注意力和FFN之后**都会进行**层归一化(LayerNorm)**, 提高训练稳定性. 

        - 6 残差连接(Residual Connection)
            - Transformer在每个子层(自注意力, 多头注意力, FFN)外都加了残差连接: 
                \[ \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x)) \]
            - 残差连接有助于信息流动和梯度传播. 

        - 7 掩码(Masking)
            - Padding Mask: 防止模型关注填充(Padding)的位置. 
            - Look-Ahead Mask(未来信息屏蔽): 用于解码器, 避免预测时看到未来信息. 

    - 3 Transformer的训练
        - Transformer的训练通常使用自回归(Auto-regressive)解码, 即在解码时, 模型只能看到当前和过去的输出.

        - 训练时, 使用交叉熵损失(Cross-Entropy Loss), 并结合Teacher Forcing技巧, 使得训练更稳定.

        - 此外, 常用的优化方法: 
            - Adam 优化器
            - 学习率调度(Learning Rate Warmup + Decay)

    - 4 Transformer的优势
        相比RNN和CNN, Transformer具有以下优势(后面有详细介绍为什么Transformer替代了LSTM)
            - 并行计算: 不像RNN需要序列化处理, Transformer可以利用矩阵计算并行处理所有词. 
            - 长距离依赖: 自注意力可以建模长距离依赖, 而RNN存在梯度消失问题. 
            - 更强的特征表达能力: 多头注意力能学习不同层次的语义关系. 

    - 5 Transformer的变体
        - Transformer架构被广泛应用, 并产生多个变体: 
            - BERT(双向编码): 适用于NLP任务(如问答, 情感分析).
            - GPT(自回归解码): 用于生成文本(如ChatGPT).
            - Vision Transformer (ViT): 用于计算机视觉任务(如图像分类). 
            - T5, BART: 用于文本生成和翻译任务. 

    - Summary
        - Transformer是一种基于*自注意力*的架构, 核心组成部分包括*多头注意力, 前馈神经网络, 残差连接, 层归一化和位置编码*, 具备*并行计算, 高效建模长距离依赖*的能力, 已经成为现代深度学习的主流架构.

- Q2 **Transformer为何使用多头注意力机制?**
    - Transformer使用多头注意力机制(Multi-Head Attention, MHA)的主要目的是增强模型的表达能力和学习不同语义关系**. 多头注意力是Transformer成功的关键之一, 使其在NLP, CV, 语音处理等多个领域取得了突破性的成果. 相比于单一注意力机制, 多头注意力具有以下几个关键优势:   

    - 1 提供多种表示信息
        - 单个注意力头只能关注输入序列中的一种关系, 比如长距离依赖或局部上下文. **多头注意力允许模型从多个不同的子空间学习不同的表示**, 从而捕捉更丰富的信息. 例如: 一个头可能关注**语法结构**(如主语与动词的关系), 另一个头可能关注**语义信息**(如代词和前文的对应关系),一个头可能关注**长距离依赖**(如翻译时保持主语一致).
        - 假设输入句子是: "The cat sat on the mat." head_1可能关注 "The" 和 "cat" 的关系(主谓), head_2可能关注 "sat" 和 "on the mat" 的关系(动作-地点), head_3可能关注 "cat" 和 "mat" 之间的长距离联系(词义相似性). 如果只有一个注意力头, 则只能捕捉到其中一种关系, 信息会丢失. 

    - 2 改善模型的泛化能力
        不同的注意力头可以关注不同的特征, 从而增强模型的鲁棒性, 提高泛化能力. 
        - 单头注意力可能过度拟合某一类模式, 而多头机制通过不同的注意力分布降低了过拟合风险. 
        - 在**不同任务(如翻译, 问答, 文本分类)中, 多头注意力可以适应不同的数据模式**, 使得模型更加通用.

    - 3 提升学习能力, 避免信息瓶颈**
        如果只使用单头注意力, 则**所有信息都必须通过一个低维投影矩阵传递**, 信息可能丢失. 而**多头注意力可以使用多个投影矩阵, 将信息投影到多个不同的子空间**, 避免单一表示的限制. 

        - 数学分析
            假设输入是维度为$d$的向量, 使用单个注意力头: 
                \[ Q, K, V \in \mathbb{R}^{d \times d} \]
            则注意力计算: 
                \[ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d}} \right) V \]
            这意味着所有注意力计算都是在同一子空间中进行的, 信息可能受限. 

            如果使用**h个头的多头注意力**, 则: 
                \[ Q_i, K_i, V_i \in \mathbb{R}^{d/h \times d/h} \quad \text{(每个头计算的投影维度降低)} \]
            然后: 
                \[ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O \]
            这样, **多个头可以在不同的子空间计算注意力**, 然后合并信息, 增强整体表达能力. 

    - 4 允许并行计算, 提高计算效率
        与RNN不同, Transformer本身就支持并行计算, 而**多头注意力机制进一步提高了计算效率**: 
        - **每个头的计算可以在不同的计算核(如GPU核)上并行执行**, 不会影响推理速度. 
        - **相比于单个大头, 多头注意力拆分计算后可以降低计算复杂度**, 同时仍然保持足够的表达能力. 

        - 计算复杂度分析
            假设序列长度为 \( n \), 单头注意力的计算复杂度是: \[ O(n^2 d) \]
            如果直接扩大单头注意力的维度, 计算复杂度会更高. 而**多头注意力拆分后, 每个头的计算量减少, 整体计算仍然在 \( O(n^2 d) \) 级别**, 但信息表达能力大幅提升. 


    - 5 增强对长距离依赖的建模能力
        对于长文本或长序列任务, 单头注意力可能会受到信息瓶颈的限制, 导致模型无法有效捕捉远程依赖. 而**多头注意力可以让不同的头专注于不同范围的依赖关系**: 
        - **某些头可以关注短距离关系**(如词法和语法结构). 
        - **某些头可以关注长距离关系**(如代词指代, 句间逻辑等). 

        这使得Transformer比RNN更擅长捕捉全局信息, 适用于长文本任务(如机器翻译, 摘要生成等). 

- Q3 **Transformer的self attention为什么Q和K使用不同的权重矩阵生成, 为何不能使用同一个值进行自身的点乘? Self-attention计算时为什么在进行softmax之前需要除以dk的平方根?**
    - 在Transformer的*self-attention*中, Q, K和V的选取和计算是基于输入的词嵌入(或先前层的输出), 并且通过不同的权重矩阵进行转换. 它们之间的差异以及为什么不能使用相同的矩阵进行计算, 主要涉及到以下几个方面: 

    - 1 Q和K的选取
        - 在Transformer的*self-attention*机制中, Q和K是通过输入向量(通常是词嵌入或者前一层的输出)与不同的权重矩阵进行线性变换获得的. 具体步骤如下: 

        - 假设输入序列为 \( X = [x_1, x_2, \dots, x_n] \), 其中每个 \( x_i \) 是一个词的嵌入表示. 
            - 查询(Query, Q): 用于表示当前词对其他词的"关注"程度. 
            - 键(Key, K): 用于表示输入序列中的每个词的特征, 查询通过与键的匹配计算注意力. 
            - 值(Value, V): 表示实际的信息内容, 最终会根据计算出的注意力权重进行加权求和. 
        - 在**self-attention**机制中, 我们通过以下公式生成Q, K和V: 
            \[ Q = X W_Q \quad K = X W_K \quad V = X W_V \]
            - 其中: 
                - \( W_Q, W_K, W_V \) 是训练的权重矩阵, 分别对应查询, 键和值的转换.
                - \( X \) 是输入序列的词嵌入或前一层的输出. 

            - 注意, 这些权重矩阵是不同的, 具体为: 
                - **\( W_Q \)** 用于将输入嵌入转换为查询向量. 
                - **\( W_K \)** 用于将输入嵌入转换为键向量. 
                - **\( W_V \)** 用于将输入嵌入转换为值向量. 

    - 2 计算过程: 
        接下来, 我们通过查询向量\( Q \) 和键向量 \( K \) 计算注意力权重. 具体计算如下: 

        \[ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V \]
        其中:
            \( Q K^T \) 计算查询向量与键向量的相似度, 得出每个词对其他词的注意力分数. 
            **softmax** 将这些相似度转换为概率分布(即注意力权重). 
            **\( \frac{1}{\sqrt{d_k}} \)** 是缩放因子, 用来避免内积值过大导致梯度消失或爆炸. 

    - 3 为什么Q和K使用不同的权重矩阵生成: 
        - 1 用途不同: 
            - **Q(查询)** 代表的是当前词想要"查询"其他词的方式, 它描述了当前词的注意力需求. 
            - **K(键)** 代表的是序列中每个词的特征, 它描述了每个词的特征可以被查询的信息. 

            通过不同的权重矩阵生成Q和K, 实际上是在生成**不同类型的特征空间**: 
                Q的权重矩阵 \( W_Q \) 会根据当前词的"意图"来映射其查询特征. 
                K的权重矩阵 \( W_K \) 会根据输入词的"全局特征"来映射其键特征. 

        - 2 增强表达能力: 
            通过分别为Q和K使用不同的权重矩阵, Transformer可以**学习到更加丰富的映射关系**, 使得模型能够在查询时关注到输入中不同的特征. Q和K是用来捕捉不同的语义信息, 因此它们的变换方式应该不同. 简单来说: 
            Q通过权重矩阵 \( W_Q \) 生成, 是对当前词"查询"能力的表达. 
            K通过权重矩阵 \( W_K \) 生成, 是对输入序列中每个词的特征的表达. 

        - 3 如果Q和K使用相同的权重矩阵: 
            如果Q和K使用相同的权重矩阵 \( W \), 那么我们会得到以下关系: 
            \[ Q = K = X W \]
            这意味着查询和键在同一空间中, 而这种情况下, Q和K之间的匹配程度仅仅依赖于输入本身的相似性, 并且不能有效地对不同的语义进行区分, 丧失了模型在特征空间中的灵活性和表达能力. **这样会限制模型的学习能力**, 无法有效地捕捉输入之间的细粒度关系.
    - 4 除以$d_k$的平方根的原因
        - 假设查询向量和键向量的每个元素是从一个标准正态分布(均值为0, 方差为1)中采样的, 当查询和键的维度 \( d_k \) 较大时, 点积 \( Q_i \cdot K_j \) 的结果大致服从一个均值为0, 方差为 \( d_k \) 的分布. 这个分布的标准差是 \( \sqrt{d_k} \), 所以我们将点积除以 \( \sqrt{d_k} \) 使得结果的方差保持在一个合理的范围内.
        - 避免点积值过大, 从而避免指数计算时的溢出或梯度爆炸.
        - 保证计算过程的数值稳定性, 使得模型能够有效学习并且训练过程稳定. 通过这种缩放, 模型能够处理较大的维度$d_k$, 同时保持训练过程中的稳定性.

    - Summary: 
        - **Q, K, V的生成**: Q, K, V分别是通过输入的词嵌入与不同的权重矩阵(\( W_Q, W_K, W_V \))线性变换得到的. 
        - **Q和K使用不同的权重矩阵**是因为它们的功能不同: Q用于表示当前词的查询意图, K用于表示输入词的特征. 不同的权重矩阵可以帮助模型从不同的空间中学习信息, 使得模型具有更强的表达能力和灵活性. 
        - **不能使用相同权重矩阵**: 如果Q和K使用相同的权重矩阵, 查询和键就会在同一特征空间中, 这会限制模型捕捉不同特征的能力, 降低模型的表达能力. 
        - 使用不同的权重矩阵生成QKV可以保证word embedding在不同空间进行投影, 增强了表达能力, 提高了泛化能力.
        - Self-attention在计算时, softmax之前需要除以dk的平方根的原因主要是: 对梯度进行scale, 缓解梯度消失的问题, dk的平方根为根据经验选择的参数.



- Q4 **transformer的并行化体现在哪里?**

- 在encoder和decoder的训练阶段可以并行训练(通过teacher-forcing和sequence mask), 但在transformer推理时无法并行, 需要单步自回归推理, 类似于RNN.

Q5 **transformer在音视频领域落地时需要注意的问题**

- 由于attention计算需要全局信息, 会造成系统的高延时以及巨大的存储开销, 需要设计chunk-wise attention, 做流式解码

Q6 **transformer中的mask机制**

- transformer中包含padding mask与sequence mask, 前者的目的是让padding(不够长补0)的部分不参与attention操作, 后者的目的是保证decoder生成当前词语的概率分布时, 只看到过去的信息, 不看到未来的信息(保证训练与测试时的一致性)

Q7 **Transformer为什么Q和K使用不同的权重矩阵生成, 为何不能使用同一个值进行自身的点乘?**

- 使用Q/K/V不相同可以保证在不同空间进行投影, 增强了表达能力, 提高了泛化能力.
- 同时, 由softmax函数的性质决定, 实质做的是一个soft版本的arg max操作, 得到的向量接近一个one-hot向量(接近程度根据这组数的数量级有所不同).如果令Q=K, 那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵, 这样self-attention就退化成一个point-wise线性映射.这样至少是违反了设计的初衷.

Q8 **Transformer计算attention的时候为何选择点乘而不是加法?两者计算复杂度和效果上有什么区别?**

- K和Q的点乘是为了得到一个attention score 矩阵, 用来对V进行提纯.K和Q使用了不同的W_k, W_Q来计算, 可以理解为是在不同空间上的投影.正因为 有了这种不同空间的投影, 增加了表达能力, 这样计算得到的attention score矩阵的泛化能力更高.
- 为了计算更快.矩阵加法在加法这一块的计算量确实简单, 但是作为一个整体计算attention的时候相当于一个隐层, 整体计算量和点积相似.在效果上来说, 从实验分析, 两者的效果和dk相关, dk越大, 加法的效果越显著.

Q9 **为什么在进行多头注意力的时候需要对每个head进行降维?**

- 将原有的高维空间转化为多个低维空间并再最后进行拼接, 形成同样维度的输出, 借此丰富特性信息

Q10 **简单介绍一下Transformer的位置编码?有什么意义和优缺点?**

- 因为self-attention是位置无关的, 无论句子的顺序是什么样的, 通过self-attention计算的token的hidden embedding都是一样的, 这显然不符合人类的思维.因此要有一个办法能够在模型中表达出一个token的位置信息, transformer使用了固定的positional encoding来表示token在句子中的绝对位置信息.

Q11 **为什么transformer使用LayerNorm而不是BatchNorm?LayerNorm 在Transformer的位置是哪里?**

- Layernorm在特征维度上做归一化, Batchnorm在样本维度做归一化.使用layernorm的原因是transformer的处理文本序列通常是变长序列, batch内数据分布有很大差异, 对样本维度做归一化意义不大, 相比较之下, 对每个序列自身特征做归一化的layernorm更稳定
- Layernorm在多头注意力层和激活函数层之间

引用: [https://zhuanlan.zhihu.com/p/363466672](https://zhuanlan.zhihu.com/p/363466672)

Q12 **简答讲一下BatchNorm技术, 以及它的优缺点**

- BN批归一化是对每一批的数据在进入激活函数前进行归一化, 可以提高收敛速度, 防止过拟合, 防止梯度消失, 增加网络对数据的敏感度

Q14 **Transformer中, Encoder端和Decoder端是如何进行交互的?**

- 通过encoder-decoder端的attention, Encoder的输出作为K, V, Q来自decoder

Q15 *Transformer中, *Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别?**

- Decoder有两层mha, encoder有一层mha, Decoder的第二层mha是为了转化输入与输出句长, Decoder的q, k和v的倒数第二个维度可以不一样, 但是encoder的qkv维度一样.
- Decoder的attention加入了sequence mask, 让模型无法看到将来的信息, 保证整体系统的因果性

Q16 **Bert的mask为何不学习transformer在attention处进行屏蔽score(sequence mask)的技巧?**

- BERT和transformer的目标不一致, bert是语言的预训练模型, 需要充分考虑上下文的关系, 而transformer需要保证其因果性, 主要考虑句子中第i个元素与前i-1个元素的关系.

Q17 **简单讲述一下wordpiece model 和BPE(byte pair encoding)方法**

- 在NLP领域传统的分词方法是使用空格分词得到固定的词汇, 带来的问题是遇到罕见的词汇无法处理即OOV(Out of Vocabulary)问题, 同时传统的分词方法也不利于模型学习词缀之间的关系或词的不同时态之间的关系.比如: walked, walking, walker之间的关系无法泛化到talked, talking, talker, word piece和BPE都属于子词方法, 解决上述问题
- Byte Pair Encoding字节对编码, 一种数据压缩方法, 将词拆分为子词, 具体为将字符串里最常见的一对连续数据字节被替换为该数据中未出现的字节, 把词的本身的意思和时态分开
- WordPiece算法可以看作是BPE的变种.不同点在于, WordPiece基于概率生成新的subword而不是下一最高频字节对

引用: [https://medium.com/towards-data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0](https://medium.com/towards-data-science/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0)

Q18 **解释self- attention, 它与其他attention的异同?**

- self-attention 可以看成一般 attention 的一种特殊情况.在 self-attention 中, 序列中的每个单词(token)和该序列中其余单词(token)进行 attention 计算, 具体到计算上即QKV矩阵均源于同源输入.self-attention 的特点在于**「无视词(token)之间的距离直接计算依赖关系, 从而能够学习到序列的内部结构」**

Q19 **Transformer 相比于 RNN/LSTM, 有什么优势?为什么?***

- RNN 系列的模型, 并行计算能力很差, RNN当前时刻的计算依赖于上一个时刻的隐层计算结果, 导致RNN的训练无法并行, 而Transformer在训练阶段, 通过sequence mask掩蔽和teacher-forcing, 可以做到并行训练
- 通过一些主流的实验证明, Transformer 的特征抽取能力比 RNN 系列的模型要好

Q20 **除了绝对位置编码技术之外, 还有哪些位置编码技术?**

- **相对位置编码**(RPE)技术, 具体又分三种: 1.在计算attention score和weighted value时各加入一个可训练的表示相对位置的参数.2.在生成多头注意力时, 把对key来说将绝对位置转换为相对query的位置3.复数域函数, 已知一个词在某个位置的词向量表示, 可以计算出它在任何位置的词向量表示.前两个方法是词向量+位置编码, 属于亡羊补牢, 复数域是生成词向量的时候即生成对应的位置信息.
- ***学习位置编码, ***学习位置编码跟生成词向量的方法相似, 对应每一个位置学得一个独立的向量

Q21 **在 BERT 中, token 分 哪3 种情况 mask, 分别的作用是什么?**

- 在 BERT 的 Masked LM 训练任务中,  会用 [MASK] token 去替换语料中 15% 的词, 然后在最后一层预测.但是下游任务中不会出现 [MASK] token, 导致预训练和 fine-tune 出现了不一致, 为了减弱不一致性给模型带来的影响, 在这被替换的 15% 语料中: 80% 的 tokens 会被替换为 [MASK] token, 10% 的 tokens 会称替换为随机的 token, 10% 的 tokens 会保持不变但需要被预测
- 第一种替换, 是 Masked LM 中的主要部分, 可以在不泄露 label 的情况下融合真双向语义信息, 
- 第二种随机替换, 因为需要在最后一层随机替换的这个 token 位去预测它真实的词, 而模型并不知道这个 token 位是被随机替换的, 就迫使模型尽量在每一个词上都学习到一个 全局语境下的表征, 因而也能够让 BERT 获得更好的语境相关的词向量(这正是解决一词多义的最重要特性), 
- 第三种的保持不变, 也就是真的有 10% 的情况下是 泄密的(占所有词的比例为15% * 10% = 1.5%), 这样能够给模型一定的 bias , 相当于是额外的奖励, 将模型对于词的表征能够拉向词的 真实表征

引用: https://www.jianshu.com/p/55b4de3de410


Q24 **简述Label Smoothing及其作用**

- 由于训练集中含有少量错误数据, label smoothing是将正样本的标签修为0.9, 负样本的标签修改为0.1(标签平滑的程度可以根据情况修改), 这是一种**soft**的学习, 也就是告诉模型, **不要这么自信**
- **Label Smoothing**可以提高神经网络的鲁棒性和泛化能力

Q25 **BERT训练时使用的学习率 warm-up 策略是怎样的?为什么要这么做?**

- warm-up相当于对学习率自适应的一个过程.因为模型一开始参数迭代的方向比较重要, 所以在开始训练的时候, 避免部分噪声样本把参数更新方向带偏, 所以开始训练的时候会设置一个warm-up步数, 当前步的学习率和当前步数成正比, 即学习率为(current_step/warm_up_step)*learning_rate.

- Q26 **为什么基于Transformer的大模型目前处于主导地位?**

    - 2018年GPT(Generative Pre-trained Transformer), BERT(Bi-Directional Encoder Representation of Transformer)在100M数量级上已经表现出了远优于LSTM等传统模型的性能, 导致研究者们大量转移至Transformer.
        - 在Transformer出来之前, 以及刚出来的那段时间, 学术界主要是在尝试LSTM(Long Short Term Memory)进行大规模预训练语言模型, 其中最出名的应该是2018年的ELMo(Embeddings from Language Model).
        - ELMo是标准的双向LSTM, 通过自回归的方式进行模型训练. 当需要应用到下游的具体任务时, 会使用额外的线性层来融合不同的token和embedding, 从而得到最终的token或者整个句子的embeddeding. 这些额外的线性层就是与下游具体任务相关的, 可学习的参数, 所以ELMo就是典型的Pre train + Fine tune模式.
        - ELMo在2018年时在6个NLP任务上最大参数量到了93.6M, 但是很快就被同年的GPT和BERT打败. GPT和BERT也是标准的Pre train + Fine Tune模式, GPT的参数量是117M, BERT-base的参数量是110M, BERT-large的参数量是340M. 所以GPT和BERT-base的参数量与ELMo相当, 但是在性能上比ELMo强不少.
        - 2018年是一个很重要的时间点, 基于Transformer的GPT和BERT因为其强大的表现, 让大量的研究者放弃了RNN系列语言模型. 所以, 在实际效果上的绝对优势是Transformer成为主流模型结构的最重要原因.

    - Transformer优于LSTM的重要原因是, Transformer能够解决长依赖的问题.
        - RNN在理论上来说, 具有捕捉长依赖的能力, 因为信息始终会沿着时间线向后传递. 但实际上, 因为基于梯度的优化方法, 在RNN的特殊结构上容易出现梯度消失, 梯度爆炸, 所以导致非常难以训练, 其表现就是远距离信息直接丢失.
        - LSTM通过引入一个cell state, 希望解决RNN的长依赖问题. 同样在2018年, 一篇名为"Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use Context"的论文分析了LSTM based LLM对于远距离token的依赖性. 这篇论文的结论很简单, 大多数LSTM based LLM只能够有效利用过去的200个token, 并且模型只对最近的50个token的位置信息敏感.
        - 引用: https://arxiv.org/pdf/1805.04623
        - 简单来说, LSTM based LLM解决不了长距离依赖问题, 因为最长也就能利用200个token左右的上下文信息.
        - 另外一种关于LSTM无法解决长距离依赖的原因是, RNN模型仅仅依靠一个低维的hidden state或cell state来存储过去所有的信息显然是不合理的.
        - 所以, transformer使用了self attention, 建立了token与token之间的"超距"关系, 所谓"超距", 指的是在transformer中根本没有距离的概念.
        - 在transformer的encoder结构中, 每个token可以attend到其他任意的token. 在transformer的decoder结构中, 每个token可以attend到这个token之前的任意一个token. 并且, 这种attention的计算过程没有任何关于位置信息的先验知识, 赋予了模型很大的自由度. 既然都没有距离的概念了, 那么"长距离"依赖的问题也就不存在了. 这是transformer优于LSTM的最重要的原因.

    - LSTM难以并行化影响了研究者将其sacle到更大规模模型的动力.
        - LSTM的难以并行化也限制了很多顶级团队的研究动力. 尤其是在transformer在1B规模上表现出了明显优于LSTM的特性之后, 大多数研究人员没有动力去做一个1B或者10B规模的LSTM based LLM.
        - 在transformer的结构中, attention本身写法比较简单, 是一个单纯的矩阵乘法, arithmetic intensity是可调的, 方便适配硬件的拓扑结构, 方便scale up.
        - Transformer的可扩展性非常好, 只要简单的堆模型, 让模型的单层并行变大, 或者把模型层数增加, 效果就会持续变好. 这个特性RNN, LSTM都不具备, 从几层到十几层, 架构都是不一样的.

- Q27 **Transformer的计算过程是怎样的?**
    - Transformer模型是近年来自然语言处理和其他序列建模任务中的突破性架构, 它采用了自注意力机制(Self-Attention), 完全摒弃了传统的循环神经网络(RNN)和卷积神经网络(CNN). Transformer的输入, 输出和计算过程主要围绕以下几个部分展开. 

    - 输入和输出
        - 输入
            - **输入序列(Input Sequence)**: Transformer接收一个长度为\(N\)的序列输入, 这个序列的每个元素是一个向量, 通常是经过嵌入(Embedding)后的词向量或特征向量. 
            - **位置编码(Positional Encoding)**: 由于Transformer没有像RNN那样的序列顺序依赖, 它通过位置编码来保留序列中元素的相对或绝对位置信息. 位置编码被加到每个输入向量上, 通常采用正弦和余弦函数进行编码. 
        
        - 输出
            - **模型输出(Output Sequence)**: Transformer的输出是一个与输入序列长度相同的序列, 每个位置的输出是通过自注意力和前馈网络的计算得出的, 通常用于序列分类, 生成, 翻译等任务. 
            - **对于解码器(Decoder)**, 其输出可以作为生成的下一个词的概率分布, 或用于下游任务的表示. 

    - Transformer的计算过程
        - Transformer的计算过程由编码器(Encoder)和解码器(Decoder)两个部分组成. 每个部分的工作都依赖于多个层的堆叠. 我们分别来看这两个部分的计算过程. 

        - 1 编码器(Encoder)计算过程: 
            编码器的目的是将输入序列转换成一组新的表示, 供解码器使用. 每个编码器层的计算过程包括两个主要部分: 自注意力机制(Self-Attention), 前馈神经网络(Feed-Forward Network).

            - (1) 自注意力机制(Self-Attention)
                自注意力机制的目标是通过关注输入序列中的不同位置来生成每个位置的表示. 它通过以下计算过程实现: 
                - 输入为一个序列的词向量, 经过位置编码后作为输入. 
                - Query (Q), Key (K), Value (V): 每个输入向量都被映射到三个不同的空间: 查询(Query), 键(Key)和值(Value). 这些映射是通过与训练得到的权重矩阵进行乘法得到的. 
                    \[ Q = XW_Q, \quad K = XW_K, \quad V = XW_V \]
                - 注意力得分计算: 接着, 计算每一对词之间的注意力得分, 使用Query和Key的点积来衡量相关性. 
                    \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
                其中, \(\sqrt{d_k}\)是对结果进行缩放的常数, 以避免点积过大. 
                - 加权求和: 经过注意力得分加权后, 得到输出的加权和. 每个位置的输出是所有位置的加权和, 即每个词向量的表示是基于所有输入位置的信息. 

            - (2) 前馈神经网络(Feed-Forward Network)
                - 自注意力计算后, 结果会通过一个前馈神经网络进行进一步的变换. 该网络通常由两个全连接层和激活函数(如ReLU)组成: 
                    \[ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \]

                - 残差连接: 每层都使用残差连接来帮助梯度流动, 从而避免梯度消失问题. 

            - (3) 层归一化(Layer Normalization)
                - 每个子层(自注意力层, 前馈网络层)之后, 都会进行层归一化以稳定训练. 

        - 2 解码器(Decoder)计算过程
            解码器的目标是根据编码器的输出生成目标序列, 通常用于生成任务(如机器翻译). 解码器的计算过程与编码器非常相似, 但也有一些区别: 

            - (1) Masked Self-Attention
                在解码器中, 第一个自注意力层是**掩蔽自注意力**(Masked Self-Attention), 即在计算注意力得分时, 不允许关注未来的词, 只能依赖已生成的词. 这是为了保证生成时的自回归性质. 
                计算方式与编码器中的自注意力相同, 只是引入了一个掩蔽矩阵, 避免“窥视”未来的词. 

            - (2) Encoder-Decoder Attention
                解码器的第二个注意力层关注编码器的输出. 这个层计算的是**编码器-解码器注意力**(Encoder-Decoder Attention), 目的是将编码器的表示与解码器的当前位置结合. 
                \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
                其中, Query来自解码器的前一层输出, Key和Value来自编码器的输出. 

            - (3) 前馈神经网络
                与编码器相同, 解码器的输出也会经过一个前馈神经网络进行变换. 

        - 3 最终输出生成
            解码器的输出会经过线性变换和Softmax激活, 得到目标词汇表的概率分布, 通常用于生成下一个词或标记. 

    - Summary
        Transformer模型的计算过程通过以下几个步骤实现: 
        1 输入序列通过嵌入和位置编码被转换为表示. 
        2 编码器通过自注意力机制和前馈神经网络生成每个输入位置的表示. 
        3 解码器生成输出序列, 结合编码器的输出, 并使用自注意力和编码器-解码器注意力机制来生成最终的输出. 
        这一过程中的核心是**自注意力机制**, 它使得Transformer能够有效地捕捉序列中远距离的依赖关系, 从而大幅提升了序列建模任务的性能.
