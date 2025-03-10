# NLP算法面试题

Q1 **NLP任务中, 文本预处理有什么作用?**

- 文本预处理会将文本转换为机器能读懂的形式, 以便机器学习算法可以更好地执行. 同时, 在情感分析等任务中, "去除停用词"等预处理有助于提高机器学习模型的准确率.
- 常见的文本预处理手段包括: 删除`HTML`标签, 删除停用词, 删除数字, 小写所有字母, 词形还原等.

Q2 **词形还原和词干提取有什么区别?**

- **词干提取**只是删除*单词的最后几个字符*, 通常会导致错误的含义和拼写, 如`eating -> eat, Caring -> Car`.
- **词形还原**考虑*上下文*并将单词转换为其有意义的基本形式, 如`Stripes -> Strip (verb) -or- Stripe (noun), better -> good`.

Q3 **何时使用词形还原?何时使用词干提取?**

- 词干提取更多被应用于信息检索领域, 如`Solr`, `Lucene`等, 用于扩展检索, 粒度较粗.
- 词形还原更主要被应用于文本挖掘, 自然语言处理, 用于更细粒度, 更为准确的文本分析和表达.
- 词干提取计算量耗费小, 适用于大数据集, 词形还原耗费计算量大, 常用于对准确性要求高的小数据集.

Q4 **`POS Tagging`(词性标注)有什么用?**

- **`POS Tagging`**用于将每个单词分类到其词性中.
- 词性可用于查找语法或词汇模式.
- 在英语中, 同一个词可以拥有不同的词性, 词性标注有助于区分它们.

Q5 **使用词袋模型(Bag of Words)提取特征有哪些优点?**

- 词袋模型(`Bag of Words`)非常简单和灵活, 可以自己设计词汇表, 可以以多种方式从文档中提取特征.
- 由相似内容组成的文本在其他方面(例如含义)也将相似, 因此词袋模型能反映句子含义间的相似性.

Q6: ***TF-IDF方法与TF方法的区别在哪里?***

- TF-IDF定义为词频-逆文档频率, TF定义为词频
- TF-IDF是一种统计方法, 旨在反映一个词对语料库集合中文档的重要性: 字词的重要性随着它在文件中出现的次数成正比增加, 但同时会随着它在语料库中出现的频率成反比下降, TF是一个词在文档中出现的次数的计数.
- TF定义为$tf_{ij}=\frac{n_{ij}}{\sum_{k}^{}n_{k,j}}$, $n_{ij}$是该词在文件中出现的次数, 分母则是文件中所有词汇出现的次数总和, TF-IDF定义为$tfidf(t,d)=tf(t,d)*log(\frac{N}{df+1})$, 其中$N$是语料库中的总文件数, $df$表示包含词语的文件数目

Q7: ***什么是one-hot vector?它们如何用于自然语言处理?***

- one-hot 向量可用于以$L*N$大小矩阵的形式表示句子, 其中$N$是语料库中单个单词的数量, $L$是句子的长度, 将词语所在下标位置置为1, 其他位置置为0

Q8: ***文本预处理有哪些方法?***

- 文本预处理主要有三个不同类型: 
- **标记化(Tokenization)**: 这是将一组文本分成更小的部分或标记的过程.段落被标记化为句子, 句子被标记化为单词.
- **规范化(Normalization)**: 数据库规范化是将数据库的结构转换为一系列规范形式.它实现的是数据的组织, 使其在所有记录和字段中看起来相似.同样, 在NLP领域, 规范化可以是将所有单词转换为小写的过程.这使得所有的句子和标记看起来都一样, 并且不会使机器学习算法复杂化.
- **去噪(Noise Removal)**: 是对文本进行清理的过程.做一些事情, 比如删除不需要的字符, 比如空格, 数字, 特殊字符等.

Q9: ***one-hot vector应用在NLP领域有什么不足之处?***

- **矩阵稀疏和维度灾难**.one-hot表示是将词语所在下标位置置为1, 其他位置置为0, 而现实生活中, 词语的集合是很大的, 达到几千甚至几万, 而每个向量的维度是和词语集合中词语的数量是一致的, 所以一个词需要用几千甚至几万的维度来表示, 如此大的维度在后续计算中需要很大的计算资源.此外, 一个向量中只有一个维度是非零的, 明显是过于稀疏的.
- **语义缺失**.在我们的表达中, 词语之间是有一定的相似性的, 例如“i”和“you”, “apple”和“banana”之间的相似性是比较高的, 而“i”和“apple”之间的相似性比比较低的.而词向量作为词语的数字特征表示, 理应需要保持词语之间语义上的相似性.但是, one-hot所得出来的每个词语的向量与其他词语的向量都是正交的, 即每个词语之间的余弦相似度均为0, 每对词语之间的欧式距离也是相同的.所以, 这种向量表示失去了词语之间的相似性.

引用: https://www.jianshu.com/p/9948c5764302

Q10: ***TF-IDF方法比TF方法好的地方有哪些?***

- TF方法只统计句子中词语出现的词频, 词语出现的越多, 重要性越高, 而TF-IDF考虑到字词的重要性会随着它在语料库中出现的频率成反比下降, 有利于分析出其中的关键词, 降低一些出现频次高但重要性低词汇的重要性, 如中文助词”的“.

Q11: ***比较TF-IDF方法和Bag of words词袋模型***

- Bag of Words 只是创建一组向量, 其中包含文档(评论)中单词出现的次数, 而 TF-IDF 模型包含有关较重要单词和较不重要单词的信息.
- 词袋模型很容易解释, 然而, TF-IDF 通常在机器学习模型中表现更好

Q12: ***解释什么是BLEU值?***

- BLEU(*bilingual evaluation understudy*)是用于评估**模型生成的句子(candidate)**和**实际句子(reference)**的差异的指标, 用于评估*机器翻译*文本的质量.BLEU实现是分别计算**candidate句**和**reference句**的**N-grams模型**, 然后统计其匹配的个数来计算得到

Q13: ***NLP处理中, 会碰到哪些歧义问题?***

- 词汇歧义: 有不止一种意思的词.
- 句法歧义: 句子中使用的语法是模棱两可的, 对于给定语法的句子, 不止一个解析树是正确的.
- 语义歧义: 一个句子的不止一种语义解释.
- 语用歧义: 当陈述不具体, 上下文没有提供澄清陈述所需的信息时, 就会出现这种情况.

Q14: ***中文分词问题中, 会碰到那些歧义问题类型?***

- 概括起来中文分词歧义主要有两种类型.分别是**交集型歧义**和**组合型歧义**
- 交集型歧义就是: ABC.既可以切分成AB/C,也可以切分成A/BC
- 组合型歧义就是: AB.既可以组合成AB, 也可以组合成A/B

引用: [https://juejin.cn/post/7175781946767704121](https://juejin.cn/post/7175781946767704121)

Q15: ***在文本情感分析任务中, 你会选用哪种loss?***

- 一个情感分析模型可以做成简单的*positive/* negative输出, 也可以有*positive/neutral/negative*输出, 也可以有一个5 stage output表示, 0表示最为消极, 4表示最为积极
- 如果输出是简单的*正/负*, 则可以使用二元**交叉熵损失.**如果输出多于`2`类别, 则可以使用**分类交叉熵**

Q16: ***解释什么是ROUGE指标?ROUGE与BLEU值有什么不同?***

- ROUGE通过将模型生成的摘要或者回答与参考答案(一般是人工生成的)进行比较计算, 得到对应的得分.ROUGE指标与BLEU指标非常类似, 均可用来衡量生成结果和标准结果的匹配程度, 不同的是ROUGE基于召回率, BLEU更看重准确率.
- ROUGE分为Rouge-N, Rouge-L, Rouge-S, Rouge-N实际上是将模型生成的结果和标准结果按N-gram拆分后, 计算召回率, Rouge-L使用了最长公共子序列, Rouge-S是Rouge-N的一种扩展, 允许跳过中间的某些词匹配

Q17: ***在机器翻译领域, 为什么encoder-decoder结构的RNN取代了seq2seq RNN?***

- seq2seq RNN一次只能翻译一个词汇, 而encoder-decoder RNN能处理变长的输入/输出, 一次网络推理能翻译一整句话

引用: [https://quizlet.com/284587332/deep-learning-midterm-2-flash-cards/](https://quizlet.com/284587332/deep-learning-midterm-2-flash-cards/)

Q18: ***解释word embedding***

- 词嵌入或词向量是表示文档和词的一种方法.它是一个数字向量, 允许具有相似含义的词具有相似的表示.word embedding可以将单词表示映射到低维度空间, 编码了语义空间信息.

Q19: ***word embedding的优点?***

- 词嵌入相比one-hot embedding, 维度低且为连续向量, 方便机器学习模型的处理
- word embedding具有天然的聚类效果, 语义相似的词在向量空间上也较为接近
- 是无监督的方法, 方便应用到海量数据

Q20: ***CNN在NLP中有哪些应用?CNN在NLP任务中应用的直觉是什么?***

- CNN最初为计算机视觉任务开发, 因此拥有平移不变性(举例来说, 们想要将一张图像分为几个类别, 例如猫, 狗, 飞机, 在这种情况下, 如果您在图像上找到一只猫, 您不关心这只猫在图像上的位置), 而在文本任务中, 短语的顺序会影响整体语义的变化.因此, 在语序对NLP任务影响不大的任务中, CNN应用的场景较多, 如情感分类等

引用: [https://dennybritz.com/posts/wildml/understanding-convolutional-neural-networks-for-nlp/](https://dennybritz.com/posts/wildml/understanding-convolutional-neural-networks-for-nlp/)

[https://lena-voita.github.io/nlp_course/models/convolutional.html](https://lena-voita.github.io/nlp_course/models/convolutional.html)

Q21: ***解释word2vec方法***

- word2vec是提取word embedding的方法之一, 是从大量文本预料中以无监督方式学习语义知识的模型.模型结构较为简单, 包括输入层, 隐藏层和输出层, 模型框架根据输入输出的不同, 主要包括**CBOW**和**Skip-gram**模型.CBOW的方式是在知道词的上下文的情况下预测当前词, 而Skip-gram是在知道了词的情况下, 对词的上下 文进行预测.

Q22: ***word2vec与Glove算法有什么相同之处和区别?***

- Glove与word2vec, 都是提取word embedding的算法, 两个模型都可以根据词汇的“co-occurrence”(语料中词汇一块出现的频率)信息, 将词汇编码成一个向量
- word2vec使用神经网络学习word-embedding的表示, 属于predictive-model, 而Glove算法本质上是对词频共现矩阵进行SVD降维, 属于count-based model
- 对于较大的数据, GloVe更容易并行化, 计算速度更快

Q23: ***解释隐马尔可夫模型(HMM)***

- 隐马尔科夫模型使用马尔科夫过程建模基于序列的问题, 模型参数$\lambda$包含初始状态分布$\pi$ , 从一种隐藏状态到另一种隐藏状态的转移概率矩阵$A$, 给定隐藏状态的观测状态概率似然矩阵$B$

![Untitled](NLP%E7%AE%97%E6%B3%95%E9%9D%A2%E8%AF%95%E9%A2%98%20e4106cc12b494ed78fdb0c8da70dd20b/Untitled.png)

- HMM中主要有两个假设: 任意时刻的隐藏状态只依赖于它前一个隐藏状态, 任意时刻的观察状态只仅仅依赖于当前时刻的隐藏状态.
- HMM包含三个问题, 评估, 学习, 解码, 
    - 评估问题给定模型$\lambda$和观察序列$O$, 计算在该模型下观察序列出现的概率, 
    - 学习问题给定观察序列$O$, 估计模型参数$\lambda$, 可以使用**Baum-Welch 算法解决**
    - 解码问题给定模型和观察序列, 求最可能出现的状态序列.可以使用Viterbi算法解决

Q24: ***bert的架构是什么 目标是什么, 输入包括了什么 三个embedding输入是怎么综合的?***

- Bert的结构主要是Transformer的encoder部分, 其中Bert_base有12层, 输出维度为768, 参数量为110M, Bert_large有24层, 输出维度为1024, 参数总量为340M.
- Bert的目标是利用大规模无标注语料训练, 获得文本包含丰富语义信息的表征.
- Bert的输入: token embedding, segment embedding, position embeddimg, 三个向量相加作为模型的输入.

Q25: ***Seq2seq模型中decode和encode的差别有哪些?***

- encoder是对输入的序列进行编码, 编码的时候不仅可以考虑当前状态, 还可以考虑前后状态, 在进行decoder的时候不能看到当前状态之后的信息, 考虑的是encoder的context vector和decoder上一个时刻的信息.

Q26: ***阐述transformer的模型架构?***

- Transformer本身是一个典型的encoder-decoder模型, Encoder端和Decoder端均有6个Block, Encoder端的Block包括两个模块, 多头self-attention模块以及一个前馈神经网络模块, Decoder端的Block包括三个模块, 多头self-attention模块, 多头Encoder-Decoder attention交互模块, 以及一个前馈神经网络模块, 需要注意: Encoder端和Decoder端中的每个模块都有残差层和Layer Normalization层.

Q26: ***bert模型有哪些可以改进的地方?***

- 问题: 中文BERT是以汉字为单位训练模型的, 而单个汉字是不一表达像词语或者短语具有的丰富语义.
- 改进: ERNIE模型, 给模型输入知识实体, 在进行mask的时候将某一个实体的汉字全都mask掉, 从而让模型学习知识实体.BERT-WWM通过短语级别遮盖(phrase-level masking)训练.
- 问题: NSP任务会破坏原本词向量的性能.
- 改进: RoBERTa, 移除了NSP任务, 同时使用了动态Mask来代替Bert的静态mask.

引用: [https://juejin.cn/post/7032813935619211295](https://juejin.cn/post/7032813935619211295)

Q27: ***讲述BPE模型***

- BPE(字节对)编码是一种简单的数据压缩形式, 用来在固定大小的词表中实现可变⻓度的子词.BPE 首先将词分成单个字符, 然后依次用另一个字符替换频率最高的**一对字符**, 直到循环次数结束..BPE方法可以有效地平衡词汇表大小和步数(编码句子所需的token数量).

Q28: ***BPE模型与Wordpiece模型有什么区别?***

- BPE与Wordpiece都是首先初始化一个小词表, 再根据一定准则将不同的子词合并.词表由小变大.
- BPE与Wordpiece的最大区别在于, 如何选择两个子词进行合并: BPE选择频数最高的相邻子词合并, 而WordPiece选择能够提升语言模型概率最大的相邻子词加入词表.

Q29: ***BERT模型里, self-attention操作里$\sqrt{d_{k}}$的作用***

- QK进行点积之后, 值之间的方差会较大, 也就是大小差距会较大, 如果直接通过Softmax操作, 会导致大的更大, 小的更小, $\sqrt{d_{k}}$是一个经验设定值, 通过除以$\sqrt{d_{k}}$进行缩放, 会使参数更平滑, 训练效果更好

Q30: ***attention机制为什么有效?***

- 在attention出现之前, 通常使用RNN(LSTM)之类的模型捕获时序信息, 这些模型读取完整的句子并将所有信息压缩为固定长度的矢量, 当处理长句时会导致信息丢失等问题
- 注意力机制是基于人类的视觉注意机制, 在处理长句时, 不是按顺序浏览每个单词或字符, 而是潜意识地关注一些信息密度最高的句子并过滤掉其余部分, 能更有效地捕获上下文信息

Q31: ***分析Bert模型存在的缺点***

- Bert无法解决长文本问题, 因为transformer模型本身是自回归的, 当token个数>512时效果会显著下降, 并且transformer模型的复杂度是$O(n^{2})$, 处理长文本时会消耗巨大的计算量
- 输入噪声 [MASK], 造成预训练 - 精调两阶段之间的差异
- 生成任务表现不佳: 预训练过程和生成过程的不一致, 导致在生成任务上效果不佳
- 位置编码使用绝对编码

Q32: ***Transformer中残差结构的作用***

- 减少梯度消失和梯度爆炸的问题, 同时能解决退化问题.退化问题是指: 当网络隐藏层变多时, 网络的准确度达到饱和然后急剧退化, 而且这个退化不是由于过拟合引起的

Q33: ***Transformer采用postnorm还是prenorm?为什么?***

- pre norm就是在残差前norm, x+F(Norm(x)), 这样残差的效果更强, 训练计算量更低, 但是会削弱模型深度带来的增益.post norm就是正常bert用的, 在残差后面加, Norm(x+F(x)), 深度带来的效果更明显, 但是计算量会更大, 目前post norm认为更适合深层transformer

Q34: ***BERT为什么用字粒度而不是用词粒度?***

- 因为在做MLM预训练任务时, 最后预测单词是用softmax进行预测.使用字粒度的话, 总字数大概在2w左右, 而使用词粒度的话, 则有十几万个词, 在训练时显存会爆炸.

Q35: ***HMM 和 CRF 算法的原理和区别?***

- HMM 是生成模型, CRF 是判别模型
- HMM 是概率有向图, CRF 是概率无向图
- HMM 求解过程可能是局部最优, CRF 可以全局最优
- HMM是做的马尔科夫假设, 而CRF是马尔科夫性, 因为马尔科夫性是是保证或者判断概率图是否为概率无向图的条件

Q36: ***BiLSTM+CRF模型中, CRF层的作用?***

- CRF 层可以为最后预测的标签添加一些约束来保证预测的标签是合法的.在训练数据训练过程中, 这些约束可以通过 CRF 层自动学习到的.
- CRF 中有转移特征, 即它会考虑输出标签之间的顺序性, 也会学习一些约束规则

Q37: ***nlp有哪些数据增强的方法?***

- 加噪声.加噪尤以去信息为主(Dropout).比如随机扔词(每次扔一类词, 每次扔一个词), 比如随机在 Embedding 上 dropout(这个几乎所有 Neural Model 都加了).有结构的 Dropout 也就是所谓的 Mask, 即使用带权的 mask 来遮盖掉一些词.
- 同义词替换. 我们可以随机的选择一些词的同义词来替换这些词, 比如: “她非常美丽” 改为 “她非常漂亮”.但是这种方法比较大的局限性在于同义词在 NLP 中通常具有比较相近的词向量, 因此对于模型来说, 并没有起到比较好的对数据增强的作用.
- 反向翻译. 这是机器翻译中一种非常常用的增强数据的方法, 主要思想就是通过机器将一个句子翻译为另一种语言, 再把另一种语言翻译为原先的语言, 得到一个意思相近但表达方式可能不同的句子.这种方法不仅有同义词替换, 词语增删的能力, 还具有对句子结构语序调整的效果, 并能保持与原句子意思相近, 是一种非常有效的数据增强方式.
- 使用生成网络.使用GAN或者VAE这些生成式网络来生成一些数据.但这种方法的难点在于需要对 GAN 模型的训练达到比较好, 才能更有效的生成高质量数据, 这一点工作量相对较大也较为复杂.

Q38: ***NLP如何解决OOV问题?***

- UNK处理, 如果训练数据充足, OOV都是冷门词汇, 可以将所有OOV词汇替换为UNK标记
- Word PIece Model: 拆词规则可以从语料中自动统计学习到, 常用的是BPE(Byte Pair Encode)编码, 
- Character Model: 把所有的OOV词, 拆成字符, 简单暴力的手段, 坏处是文本序列变得非常长, 对于性能敏感的系统, 这是难以接受的维度增长
- 扩大词表

Q39: ***分析Bert, 不同层针对NLP的什么任务?***

- **POS, 成分分析, DEPS, Entities, SRL, COREF, 关系分类, 从上到下, 越往下越偏向高层语义的知识.**POS 词性标注是简单任务, 偏向表层特征, 关系分类则是纯语义的任务, 不理解语义便无法很好的解决任务, 从上到下逐渐趋向语义任务.

引用: [https://cloud.tencent.com/developer/article/1526735](https://cloud.tencent.com/developer/article/1526735)

Q40: ***Albert里的SOP为什么会有效?***

- ALBERT 认为, NSP (下一个句子预测) 将话题预测和连贯预测混为一谈.作为参考, NSP 使用了两个句子 —— 正样本匹配是第二个句子来自同一个文档, 负样本匹配是第二个句子来自另一个文档.相比之下, ALBERT 的作者认为句子间的连贯是真正需要关注的任务 / 损失, 而不是主题预测, 因此 SOP 是这样做的: 使用了两个句子, 都来自同一个文档.正样本测试用例是这两句话的顺序是正确的.负样本是两个句子的顺序颠倒.

Q41: ***BERT的三个embedding为什么可以相加?***

- 三个embedding分别为position embedding, token embedding, segment embedding, 将三个特征相加时DL中通用的特征交互的方法, 通过特征的交叉, 分别是token, position, segment.高阶的交叉带来更强的个性化表达能力, 即带来丰富语义的变化.规避了transformer因为位置信息丢失造成的上下文语义感知能力.

Q42: **Word2vec和LDA两个模型有什么区别和联系?**

- LDA利用文档中单词的共现关系来对单词按主题聚类, 也可以理解为对“文档-单词”矩阵进行分解, 得到“文档-主题”和“主题-单词”两个概率分布, 而 Word2Vec其实是对“上下文-单词”矩阵进行学习, 其中上下文由周围的几个单词组成, 由此得到的刺向了表示更多地融入了上下文共现的特征, 也就是说, 如果两个单词所对应的Word2Vec向量相似度高, 那么他们很可能经常在相同的上下文中出现.
- 主题模型和词嵌入两类方法最大的不同其实在于模型本身, 主题模型是一种基于概率图模型的生成式模型, 其似然函数可以写成若干条件概率连乘的形式, 其中包括需要推测的隐含变量(即主题), 而词嵌入模型一般表达为神经网络的形式, 似然函数定义在网络的输出之上, 需要通过学习网络的权重得到单词的稠密向量表示.

Q43: **相对位置编码和绝对位置编码有什么区别?**

- 绝对位置编码如三角函数位置编码, BERT, GPT也用了绝对位置编码, 但是他们采用的是可学习的绝对位置编码.但绝对位置编码有两大缺陷: 
    - **三角函数绝对位置编码只考虑距离没有考虑方向**
    - **距离表达在向量project以后也会消失**
- 相对位置编码的目的是建模位置的**相对含义**, 如Sinusoidal Position Encoding和Complex embedding
    - Sinusoidal Position Encoding使用正余弦函数表示绝对位置, 通过两者乘积得到相对位置
    - Complex embedding使用了复数域的连续函数来编码词在不同位置的表示

Q44: **Elmo 的思想是什么?**

- 预训练时, 使用语言模型学习一个单词的emb(**多义词无法解决**), 
- 使用时, 单词间具有特定上下文, 可根据上下文单词语义调整单词的emb表示(**可解决多义词问题**)
    - 理解: 因为预训练过程中, emlo 中 的 lstm 能够学习到 每个词 对应的 上下文信息, 并保存在网络中, 在 fine-turning 时, 下游任务 能够对 该 网络进行 fine-turning, 使其 学习到新特征, 

Q45: **word2vec中霍夫曼树是什么?**

- HS用哈夫曼树, 把预测one-hot编码改成预测一组01编码, 进行层次分类.
    - 输入输出: 
        - 输入: 权值为(w1,w2,...wn)的n个节点
        - 输出: 对应的霍夫曼树
- 步骤: 
    1. 将(w1,w2,...wn)看做是有n棵树的森林, 每个树仅有一个节点.
    2. 在森林中选择根节点权值最小的两棵树进行合并, 得到一个新的树, 这两颗树分布作为新树的左右子树.新树的根节点权重为左右子树的根节点权重之和.
    3. 将之前的根节点权值最小的两棵树从森林删除, 并把新树加入森林.
    4. 重复步骤2)和3)直到森林里只有一棵树为止.
    

Q46: **Word2vec 中 为什么要使用霍夫曼树?**

- 一般得到霍夫曼树后我们会对叶子节点进行霍夫曼编码, 由于权重高的叶子节点越靠近根节点, 而权重低的叶子节点会远离根节点, 这样我们的高权重节点编码值较短, 而低权重值编码值较长.这保证的树的带权路径最短, 也符合我们的信息论, 即我们希望越常用的词拥有更短的编码.如何编码呢?一般对于一个霍夫曼树的节点(根节点除外), 可以约定左子树编码为0, 右子树编码为1.在word2vec中, 约定编码方式和上面的例子相反, 即约定左子树编码为1, 右子树编码为0, 同时约定左子树的权重不小于右子树的权重.
- 使用霍夫曼二叉树, 之前计算量为$V$,现在变成了$log_{2}V$

Q47: **word2vec和NNLM对比有什么区别?**

- NNLM: 是神经网络语言模型, 使用前 n - 1 个单词预测第 n 个单词;
- word2vec : 使用第 n - 1 个单词预测第 n 个单词的神经网络模型.但是 word2vec 更专注于它的中间产物词向量, 所以在计算上做了大量的优化.优化如下: 
    - 对输入的词向量直接按列求和, 再按列求平均.这样的话, 输入的多个词向量就变成了一个词向量.
    - 采用分层的 softmax(hierarchical softmax), 实质上是一棵哈夫曼树.
    - 采用负采样, 从所有的单词中采样出指定数量的单词, 而不需要使用全部的单词
    

Q48: **word2vec和tf-idf 在相似度计算时的区别?**

1. word2vec 是稠密的向量, 而 tf-idf 则是稀疏的向量, 
2. word2vec 的向量维度一般远比 tf-idf 的向量维度小得多, 故而在计算时更快, 
3. word2vec 的向量可以表达语义信息, 但是 tf-idf 的向量不可以, 
4. word2vec 可以通过计算余弦相似度来得出两个向量的相似度, 但是 tf-idf 不可以, 

Q49: **word2vec负采样有什么作用?**

- 可以大大降低计算量, 加快模型训练时间
- 保证模型训练效果, 因为目标词只跟相近的词有关, 没有必要使用全部的单词作为负例, 来更新它们的权重

Q50: **elmo, GPT, bert三者之间有什么区别?**

- elmo 是一种深度语境化词表征模型, 它先通过大规模文本训练一个双向的(此处是用两个单向的 LSTM 模型的输出结构进行拼接, 跟一般的双向 LSTM 的定义不太一样)语言模型, 即使用上一个词预测下一个词(当然这里跟 word2vec 不一样的地方在于, 输入的是一个句子, 预测的也是一个句子, 只是本质上是用上一个单词预测下一个单词).在使用词向量的时候, 将目标单词所在的整个句子输入 ELMO 中, 然后对求得的句子向量按列求和, 就能得到该目标单词的词向量, 且该词向量能够表达一词多义的问题.
- GPT 是一个使用 Transformer 模型的 Decoder 部分进行预训练的模型, 它通过大规模文本训练一个生成式的模型的方式, 来训练词向量.GPT 不是双向的, 因为在训练的时候, 当前节点的后面节点的输入是会被遮蔽(mask)的.
- BERT 是一个使用 Transformer 模型的 Encoder 部分进行预训练的模型, 它通过大规模文本训练一个生成式的模型的方式, 来训练词向量.但是它通过对输入进行遮蔽(mask)的方法, 使得 BERT 能够使用左向和右向的双向神经网络层, 因为此时的目标词已经被遮蔽了, 所以不必担心双向神经网络层会泄露目标词.

引用: [https://github.com/km1994/NLP-Interview-Notes/tree/main/NLPinterview](https://github.com/km1994/NLP-Interview-Notes/tree/main/NLPinterview)

[https://github.com/laddie132/NLP-Interview](https://github.com/laddie132/NLP-Interview)
