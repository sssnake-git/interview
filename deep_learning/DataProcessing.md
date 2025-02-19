# data processing

Q1 特征工程与特征选择有什么区别?

- 特征工程允许我们从已有的特征中创建新的特征, 以帮助机器学习模型做出更有效和准确的预测.特征工程的任务是: 
    - 填充变量中的缺失值.
    - 将分类变量编码为数字.
    - 变量转换.
    - 从数据集中可用的特征中创建或提取新特征.
- 特征选择允许我们从特征池中选择特征, 有助于机器学习模型更有效地对目标变量进行预测.
- 典型的机器学习流水线中, 我们在完成特征工程后进行特征选择

Q2: **协方差和相关性有什么区别?**

- 协方差衡量一个变量的变化是否导致另一个变量的变化, 并处理数据集中仅变量的线性关系.它的值范围从负无穷到正无穷.简单的说, 协方差表示变量之间线性关系的方向

![Untitled](data%20processing%207d9302ed83864e1db8d51f974f19eff8/Untitled.png)

- 相关性衡量两个或多个变量彼此相关的强度, 它的值介于-1到1之间.相关性也衡量两个变量之间线性关系的强度和方向, 是协方差的函数.

Q3: **你会在大型数据集使用K-NN吗?为什么**

- 不建议在大型数据集上执行**K-NN**, 因为计算和内存成本会增加, 
- KNN的计算流程
    1. 首先计算训练集中所有向量的距离并存储它们.
    2. 对计算出的距离进行排序.
    3. 存储 K 个最近的向量.
    4. 计算出 K 个最近向量显示的最频繁的类.

Q4: **简述交叉验证方法**

- 交叉验证, 就是重复的使用数据, 把得到的样本数据进行切分, 组合为不同的训练集和测试集, 用训练集来训练模型, 用测试集来评估模型预测的好坏
- 交叉验证用在数据不是很充足的时候

![Untitled](data%20processing%207d9302ed83864e1db8d51f974f19eff8/Untitled%201.png)

Q5: **解释你是如何理解降维的**

- 降维是指降低数据的维度, 使用更少更具辨别力的特征, 可以描述数据中的大部分方差, 从而保留大部分相关信息
- 有多种技术可用于执行此操作, 包括但不限于PCA, ICA和Matrix Feature Factorization.

Q6: **解释Normalizing和scaling的区别**

- Normalizing是指归一化, 会改变数据的分布情况, 使转换后的数据大致呈正态分布
- scaling是指将数据乘以一个常数, 不改变数据分布情况

Q7: **什么情况下, 更少的训练数据会提高更高的模型准确性?**

- 从数据中删除冗余数据, 例如: 外观相似的图像, 语义类似的句子

Q8: **如何处理不平衡的数据?**

- 不平衡的数据是指: 其中一个或几个标签占数据集的大部分, 而其他标签的示例则少得多
- 常用的平衡方法有: 
    - 下采样: 减少模型训练期间使用的多数类的示例数量.通常与 Ensemble 模式结合以获得更好的效果.
    - 上采样: 通过复制少数类示例和生成额外的合成示例来增加少数类数量.
    
    ![Untitled](data%20processing%207d9302ed83864e1db8d51f974f19eff8/Untitled%202.png)
    
    - 加权类别: 通过加权类别, 告诉模型在训练期间更重视少数标签类别

Q9: **特征插补值有哪些选择?**

- 对于数字特征: 
    - 如果数据呈正态分布, 则使用平均值.
    - 如果数据倾斜或有很多异常值, 请使用中值.
- 对于分类特征: 
    - 如果数据是可排序的, 则使用中值.
    - 如果数据不可排序, 请使用众值.

Q10: **简述K折交叉验证的工作流程**

- 将原始数据分成不相交的K组(K-Fold), 每个子集数据分别做一次验证集, 其余的K-1组子集数据作为训练集, 这样会得到K个模型.这K个模型分别在验证集中评估结果, 最后的误差MSE(Mean Squared Error)加和平均就得到交叉验证误差.交叉验证有效利用了有限的数据, 并且评估结果能够尽可能接近模型在测试集上的表现, 可以做为模型优化的指标使用.

Q11: **为什么数据在高维空间更稀疏?**

- 高维空间的体积更大, 而数据点的数量永远无法与该体积相提并论, 因此高维数据通常非常稀疏

Q12: ***Bagging*和*Boosting*算法有什么区别?**

- **Bagging**: 训练集是在原始集中有放回选取的, 从原始集中选出的各轮训练集之间是独立的.
- **Boosting**: 每一轮的训练集不变, 只是训练集中每个样例在分类器中的权重发生变化. 而权值是根据上一轮的分类结果进行调整

Q13: **使用决策树做模型有哪些缺点?如何解决?**

- 决策树的缺点: 
    - 随着记录数量的增加, 决策树的训练需要大量时间, 时间复杂度非常大.
    - 不能用于大数据, 如果数据量太大, 那么一棵树可能会长出很多节点, 这可能会导致过高的复杂性并导致过拟合
- 解决方案: 
    - 对决策树进行剪枝, 缓解过拟合现象
    - 使用随机森林, 使用多棵树的集成学习

Q14: **在神经网络训练中, 如何选择数据缩放scaling方法?**

- Min-max归一化, 保留分数的原始分布, 将所有分数转换为在 [0, 1]范围内.然而, 这种方法对异常值高度敏感
- 数据标准化: 最常用的技术, 使用给定数据的算术平均值和标准差计算
- 中位数和MAD: 中位数和中位数绝对偏差 (MAD) 对离群值和分布极端尾部的点不敏感, 因此它很健壮.但是, 该技术不保留输入分布, 也不将分数转换为统一的数值范围