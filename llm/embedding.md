# Embedding

- **What is embedding?**
    - 将离散的输入id转为连续的向量表示, 通过查表为每个符号赋予带有语义信息的向量.
    - 大模型的输入是许多text, 输入的text会经过一个tokenizer, tokenzier将输入变成一个个词元(token), 具体形式是一个个id.
    - 之后会把token作为输入, 输入到embedding层, 

- **Absolute Position Encoding**
    - 