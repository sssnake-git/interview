# ASR

Q1 **Conformer模型结构**

- Conformer的模型在输入时首先对输入通过卷积进行下采样, 即通过一个卷积核大小为3x3, 步长为2的二维卷积层, 将输入的维度降为原先的1/4, 后续通过一系列的Conformer Blocks模块, 代替以往的Transformer Blocks进行语音识别任务.
- 一个Conformer Block主要包含三个基本模块: 前馈神经网络模块(feed forward module, FFM), 多头自注意力机制模块(multi-head self attention module, MHSAM)和卷积模块(convolution module, CM), 以在多头自注意力机制模块与卷积模块两侧各连接一个前馈神经网络模块为具体结构.这种结构将Transformer中的一个前馈神经网络模块拆分成两个半步前馈层连接注意力层的输出与输入, 能够比使用单一前馈层更好地提升模型效果.

Q2 **Transformer如何实现流式识别**

- 基于块(`chunk`)的解决方案

第一种方案为基于块的解决方案, 如下图所示, 其中虚线为块的分隔符. 其主体思想是把相邻帧变成一个块, 之后根据块进行处理. 这种方法的优点是可以快速地进行训练和解码, 但由于块之间没有联系, 所以导致模型准确率不佳.

![Untitled](ASR/Untitled.png)

- 基于块的流式`Transformer`

首先, 在当前帧左侧, 让每一层的每一帧分别向前看$n$帧, 这样随着层数的叠加, `Transformer`对历史的视野可以积累. 如果有$m$层, 则可以看到历史的$n \times m$帧.虽然看到了额外的$n \times m$帧, 但这些帧并不会拖慢解码的时间, 因为它们的表示会在历史计算时计算好, 并不会在当前块进行重复计算. 与解码时间相关的, 只有每一层可以看到帧的数目.

因为希望未来的信息延时较小, 所以要避免层数对视野的累积效应. 为了达到这个目的, 可以让每一个块最右边的帧没有任何对未来的视野, 而块内的帧可以互相看到, 这样便可以阻止延时随着层数而增加.

![Untitled](ASR/Untitled1.png)

Q3 **`Multi-Head Attention`中多头的作用**

- 多头的注意力有助于网络捕捉到更丰富的特征信息, 允许模型在不同位置共同关注来自不同表示子空间的信息. 类似于`CNN`中的多个卷积核.

Q4 **解释空洞卷积, 空洞卷积有什么问题?**

- 在卷积核中增加空洞来增加感受野, 不增加过多计算. 普通卷积有着3*3的卷积核空洞卷积有着3*3的卷积核, 空洞rate为2, 可以使得神经网络在同样的层数下, 拥有更大的感受野.
- 空洞卷积的卷积核不连续, 不是所有的信息参与了计算, 导致信息连续性的损失, 引起栅格效应.

Q5 **`Subword`建模对集外词的拓展?**

- 利用subword拓展skip-gram模型意思即为使用单词的一部分来代替单词, 这样的好处是可以利用词根, 词缀之类的结构来增大信息量, 强化对于不常见词的识别.
- 举例说明: 现有一句话"the quick brown fox jumped over the lazy dog", 按照原本的skip-gram模型, 给定单词quick, 模型需要学习通过quick预测到单词the和brown.subword扩展将单词粒度细化, quick变成了{qui, uic, ick}, 给定子词uic, 模型需要学习通过uic预测到子词qui和ick

Q6 **`CTC`的原理**

- `CTC`(`Connectionist Temporal Classification`)是一种避开输入与输出手动对齐的一种方式, 用来解决输入和输出序列长度不一, 无法对齐的问题.
- `CTC`算法对于一个给定的输入$X$, 它可以计算对应所有可能的输出$Y$的概率分布. 通过该概率分布, 可以预测最大概率对应的输出或者某个特定输出的概率.
- `CTC`通过引入$blank$标志, 解决输入和输出之间的对齐问题, 通过动态规划算法求解最优序列.

- 引用: [https://zhuanlan.zhihu.com/p/568176479](https://zhuanlan.zhihu.com/p/568176479)

Q7 **`CTC`有哪几个特点?**

- 条件独立: `CTC`的一个非常不合理的假设是其假设每个时间片都是相互独立的, 这是一个非常不好的假设. 在`OCR`或者语音识别中, 各个时间片之间是含有一些语义信息的, 所以如果能够在CTC中加入语言模型的话效果应该会有提升.
- 单调对齐: `CTC`的另外一个约束是输入与输出之间的单调对齐, 在`OCR`和语音识别中, 这种约束是成立的. 但是在一些场景中例如机器翻译, 这个约束便无效了.
- 多对一映射: `CTC`的又一个约束是输入序列的长度大于标签数据的长度, 但是对于标签数据的长度大于输入的长度的场景, `CTC`便失效了.

Q8 **`CTC`和端到端的关系**

- 目前端到端的语音识别方法主要有基于`CTC`和基于`Attention`两类方法及其改进方法. `CTC`实质是一种损失函数, 常与`LSTM`联合使用. 基于`CTC`的模型结构简单, 可读性较强, 但对发音字典和语言模型的依赖性较强, 且需要做独立性假设. `RNN-Transducer`模型对`CTC`进行改进, 加入一个语言模型预测网络, 并和`CTC`网络通过一层全连接层得到新的输出, 这样解决了`CTC`输出需做条件独立性假设的问题.

Q9 **`CTC`和`chain`的关系**

- 相同点: 都是在`label`生成的多条特定路径后的前向后向运算.
- 区别
  - `label`生成的多条特定路径的方式不同: `chain numerator`的多条特定路径是对齐产生的, `CTC`是通过下图方式得到.
  - blank label不同: chain numerator 没有blank label,如果认为triphone三个state 中的最后一个state 是blank label的话, 那么chain numerator的每个triphone都有自己的blank label, 在ctc中所有的phone 共享一个blank label.
  - 是否考虑了language model: 在chain numerator计算前向/后向概率时, 每一个跳转的概率都有自己的权重.在ctc中, 每一步跳转都认为是等概率的,直接忽视跳转概率.
  - 训练样本的时长不同由于chain numerator是alignment的路径, 所以长度可以控制.无论输入是多少, 最终都可以截断成150frames的训练数据.而ctc是把整个样本全部拿来训练.

![Untitled](ASR/Untitled2.png)

Q10 **如何理解强制对齐?**

- 强制对齐(`Forced Alignment`), 是指给定音频和文本, 确定每个单词(音素)的起止位置的过程, 一般使用`Viterbi`解码实现. 强制对齐是语音识别的一种特殊的, 简化了的情况, 由于它的简单性, 强制对齐通常具有较高的准确率(音素级别准确率可达`90%`, 单词级别可达`95%`以上). 使用强制对齐, 我们就可以对我们收集到的标准发音的数据进行处理: 根据音频和它对应的文本进行强制对齐, 得到每个音标对应的片段, 对各个音标收集到的样本抽取特征并进行训练. 通过对大量数据进行强制对齐, 我们对每个音标得到一个模型, 该模型将用于后续的打分流程.

Q11 **`CTC`与语言模型的结合**

- 在一些语言识别问题中, 也会有人在输出中加入语言模型来提高准确率, 所以我们也可以把语言模型作为推理的一个考虑因素: 

![Untitled](ASR/Untitled3.png)

L(Y)以语言模型token为单位, 计算Y的长度, 起到单词插入奖励的作用.如果 L(Y)是一个基于单词的语言模型, 那它计数的是Y中的单词数量, 如果是一个基于字符的语言模型, 那它计数的就是Y中的字符数.语言模型的一个突出特点是它只为单词/字符扩展前的形式计分, 不会统计算法每一步扩展时的输出, 这一点有利于短前缀词汇的搜索, 因为它们在形式上更稳定.集束搜索可以添加语言模型得分和单词插入项奖励, 当提出扩展字符时, 我们可以在给定前缀下为新字符添加语言模型评分.

Q12 **解释BPE训练准则**

- 字节对编码(BPE, Byte Pair Encoder), 又称 digram coding 双字母组合编码, 是一种数据压缩算法, 用来在固定大小的词表中实现可变长度的子词.该算法简单有效, 因而目前它是最流行的方法.BPE的训练和解码范围都是一个词的范围.BPE 首先将词分成单个字符, 然后依次用另一个字符替换频率最高的一对字符 , 直到循环次数结束.

Q13 解释FST压缩算法, FST有什么优点?**

- FST是一种有限状态转移机, 有两个优点: 1)空间占用小.通过对词典中单词前缀和后缀的重复利用, 压缩了存储空间, 2)查询速度快.O(len(str))的查询时间复杂度.

Q14 **`HMM`前向算法与`Viterbi`算法的区别**

- 隐马尔科夫模型用来解决三个问题: 评估, 解码和学习. 
- 在隐马尔科夫模型中, 前向算法是用来评估, 即计算某个观察序列的概率, 而维特比算法则是用来解码, 即寻找某个观察序列最佳的隐藏状态序列. 除此之外, 维特比算法的目的是找出当前时刻, 指定观察值时, 哪个状态的前向概率最大, 并记下这一状态的前一状态, 递归求得最优路径, 而前向算法则是保存当前的前向概率, 作为计算下一时刻前向概率的输入值, 最终求序列所有前向概率和得到观察序列的概率.

Q15 **解释`CTC`对齐, `HMM`对齐, `RNN-T`对齐异同**

- 对齐方式: `HMM`对齐是对`token`进行重复, `CTC`对齐是引入了`blank`, 可以插在任何地方, 但是其个数和`token`重复个数的和, 要等于`acoustic features`的个数, `RNN-T`对齐也引入了`blank`, 但是其是作为一个`acoustic feature`结束, 下一个`acoustic features`开始的间隔, 因此, `blank`的个数就等于`acoustic features`的个数.

![Untitled](ASR/Untitled4.png)

Q16 **`beam search`的算法细节与具体实现**

- 算法原理: beam search有一个超参数beam_size, 设为 k .第一个时间步长, 选取当前条件概率最大的 k 个词, 当做候选输出序列的第一个词.之后的每个时间步长, 基于上个步长的输出序列, 挑选出所有组合中条件概率最大的 k 个, 作为该时间步长下的候选输出序列.始终保持 k 个候选.最后从k 个候选中挑出最优的.
- 中心思想: 假设有n句话, 每句话的长度为T.$encoder$的输出shape为(n, T, hidden_dim), 扩展成(n*beam_size, T, hidden_dim).decoder第一次输入shape为(n, 1), 扩展到(n*beam_size, 1).经过一次解码, 输出得分的shape为(n*beam_size, vocab_size), 路径得分log_prob的shape为(n*beam_size, 1), 两者相加得到当前帧的路径得分.reshape到(n, beam_size*vocab_size), 取topk(beam_size), 得到排序后的索引(n, beam_size), 索引除以vocab_size, 得到的是每句话的beam_id, 用来获取当前路径前一个字, 对vocab_size取余, 得到的是每句话的token_id, 用来获取当前路径下一字.

Q17 **`TTS`端到端模型与传统模型的区别**

- 传统模型: 传统模型通常由不同的组件组成, 例如文本处理模块, 声学模型, 声码器等等. 一方面不同的组件之间相互组装设计比较费力, 另一方面由于组件之间单独训练, 可能会到导致每个组成部分之间的错误叠加, 从而不断放大误差.
例如统计参数语音合成(`TTS`)中通常有提取各种语言特征的文本前端, 持续时间模型, 声学特征预测模型等等. 这些组件基于广泛的领域专业知识, 并且设计起来很费力. 它们也是独立训练的, 所以每个组成部分的错误可能会叠加. 现代`TTS`设计的复杂性导致在构建新系统时需要大量的工程工作.
- 端到端模型: 端到端模型首先减轻了费力的组装设计, 利用一个神经网络代替了传统模型中复杂的建模过程, 其次更容易对各种(例如音色或者语种)属性或者高级特征(例如语音中的情感)进行特征的捕获与提取. 单个模型相比于每个组件错误累加来说更加健壮, 能有效减少错误的积累.

Q18 **`mfcc`特征与`fbank`的区别**

- `fbank`只是缺少`mfcc`特征提取的`dct`倒谱环节, 其他步骤相同.
- `fbank`的不足: `fbank`特征已经很贴近人耳的响应特性, 但是仍有一些不足, 其相邻的特征高度相关(相邻滤波器组有重叠), 因此当我们用`HMM`对音素建模的时候, 几乎总需要首先进行倒谱转换, 通过这样得到`mfcc`特征.
- 计算量: `mfcc`是在`fbank`的基础上进行的, 所以`mfcc`的计算量更大.
- 特征区分度: `fbank`特征相关性较高, `mfcc`具有更好的判别度, 所以大多数语音识别论文中用的是`mfcc`, 而不是`fbank`.

Q19 **简述如何实现单音素模型到三音素模型的对齐**

- `Mono phone`模型的假设是一个音素的实际发音与其左右的音素无关. 这个假设与实际并不符合. 由于`Mono phone`模型过于简单, 识别结果不能达到最好, 因此需要继续优化升级. 就此引入多音素的模型, 最为熟悉的就是`Tri phone`模型, 即上下文相关的声学模型. 通过`Mono phone`模型得到`Mono phone`各个状态所有对应的特征集合, 然后做上下文的扩展得到`Tri phone`的各个状态所有对应的特征集合, 我们可以把训练数据上所有的`Mono phone`的数据对齐转为`Tri phone`的对齐.

![Untitled](ASR/Untitled5.png)

Q20 **`ASR`三音素模型中, 为什么要做状态绑定?**

- 假如有`218`音素, 若使用`Tri phone`模型则有$218^3$个`Tri phone`. 如果不进行聚类, 需要建立$218 \times 218 \times 218 \times 3$个混合`gmm`模型(假设每个`triphone`有`3`个状态). 一方面计算量巨大, 另一方面会引起数据稀疏. 所以会根据数据特征对三音子的状态进行绑定.

Q21 **`EM`算法为什么会陷入局部最优? `EM`算法最终会收敛吗?**

- `EM`算法的似然函数是非凸的, 因此会陷入局部最优.
- `E-Step`主要通过观察数据和现有模型来估计参数, 然后用这个估计的参数值来计算似然函数的期望值, 而`M-Step`是寻找似然函数最大化时对应的参数. 由于算法会保证在每次迭代之后似然函数都会增加, 所以函数最终会收敛.

Q22 **`HMM`的特点**

- HMM是隐马尔科夫过程, 对语音信号的时间序列结构建立统计模型. HMM 包含隐藏状态序列和输出状态序列, 前者通过后者表现出来, 状态之间的转移通过概率转移矩阵来建模.人的言语过程实际上就是一个双重随机过程, 语音信号本身是一个可观测的时变序列, 是由大脑根据语法知识和言语需要(不可观测的状态)发出的音素的参数流.可见, HMM合理地模仿了这-一过程, 很好地描述了语音信号的整体平稳性和局部平稳性, 是较为理想的一种语音模型.

Q23 **`GMM`与`k-means`的关系**

- K-Means算法可以看作是一种简化的混合高斯模型, 在GMM模型中, 需要估计的参数有每个高斯成分前的系数, 每个高斯成分的协方差矩阵和均值向量.K-Means等价于固定GMM中每个高斯成分的系数都相等, 每个高斯成分都协方差矩阵为单位阵, 只需要优化每个高斯成分的均值向量.

Q24 **`RNN-T`模型结构**

- `RNN-T`在`CTC`模型的`Encoder`基础上, 又加入了将之前的输出作为输入的一个RNN, 称为Prediction Network, 再将其输出的隐藏向量与encoder得到的输出放到一个joint network中, 得到输出logit再将其传到softmax layer得到对应的class的概率.

![Untitled](ASR/Untitled%206.png)

Q25 **模型的加速方法**

(1)轻量化模型设计

从模型设计时就采用一些轻量化的思想, 例如采用深度可分离卷积, 分组卷积等轻量卷积方式, 减少卷积过程的计算量.此外, 利用全局池化来取代全连接层, 利用1×1卷积实现特征的通道降维, 也可以降低模型的计算量, 这两点在众多网络中已经得到了应用.

对于轻量化的网络设计, 目前较为流行的有SqueezeNet, MobileNet及ShuffleNet等结构.其中, SqueezeNet采用了精心设计的压缩再扩展的结构, MobileNet使用了效率更高的深度可分离卷积, 而ShuffleNet提出了通道混洗的操作, 进一步降低了模型的计算量.

(2)BN层合并

在训练检测模型时, BN层可以有效加速收敛, 并在一定程度上防止模型的过拟合, 但在前向测试时, BN层的存在也增加了多余的计算量.由于测试时BN层的参数已经固定, 因此可以在测试时将BN层的计算合并到卷积层, 从而减少计算量, 实现模型加速.

(3)网络剪枝

网络剪枝: 在卷积网络成千上万的权重中, 存在着大量接近于0的参数, 这些属于冗余参数, 去掉后模型也可以基本达到相同的表达能力, 因此有众多学者以此为出发点, 搜索网络中的冗余卷积核, 将网络稀疏化, 称之为网络剪枝.具体来讲, 网络剪枝有训练中稀疏与训练后剪枝两种方法.

(4)权重量化

是指将网络中高精度的参数量化为低精度的参数, 从而加速计算的方法.高精度的模型参数拥有更大的动态变化范围, 能够表达更丰富的参数空间, 因此在训练中通常使用32位浮点数(单精度)作为网络参数的模型.训练完成后为了减小模型大小, 通常可以将32位浮点数量化为16位浮点数的半精度, 甚至是int8的整型, 0与1的二值类型.典型方法如Deep Compression.

(5)张量分解

由于原始网络参数中存在大量的冗余, 除了剪枝的方法以外, 我们还可以利用SVD分解和PQ分解等方法, 将原始张量分解为低秩的若干张量, 以减少卷积的计算量, 提升前向速度.

(6)知识蒸馏

通常来讲, 大的模型拥有更强的拟合与泛化能力, 而小模型的拟合能力较弱, 并且容易出现过拟合.因此, 我们可以使用大的模型指导小模型的训练, 保留大模型的有效信息, 实现知识的蒸馏.

引用: [https://zhuanlan.zhihu.com/p/575065661](https://zhuanlan.zhihu.com/p/575065661)
