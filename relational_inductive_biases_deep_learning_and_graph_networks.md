本文由DeepMind，Google Brain，MIT等机构的27位作者于2018年联合发表，文章的一大部分是对之前工作的整理和回顾，并提出了
graph-to-graph的Graph Networks，which intend to incorporate deep learning methods and relational reasoning,
and is hopeful to solve the problem of relational reasoning in deep learning. 以下开始是论文读书笔记：
## 0 abstract
极易获取的数据和较低的计算代价 十分契合 deep learning methods的天然优势，使得AI在诸如视觉、语言、控制等领域取得了很大的进步。
但是，人工智能的很多defining characteristics，目前的方法无法实现，**具体来讲，在本文中作者指的是：人类将已知经验/知识推广到未知事物上的
组合泛化能力**。

论文主张：为了使AI具有类人智能，必须把combinatorial generalization作为首要优先的目标，实现这个目标的关键在于：结构化表示和结构化计算。
（在本文中，这两个名词其实指的就是graphs以及computations over graphs，后续论述中我们将结合具体的任务和实现方式来获得关于它们的直观理解）。
本文反对将hand-engineering和end-to-end learning对立起来，主张将二者结合并互补信息中受益。
+ 本文探索了 如何在深度学习框架中使用relational inductive biases 来促进学习 entities，relations，and rules for composing them；

+ 提出一个带有strong relational inductive bias 的新的building block，即graph network，它推广和扩展了各种对图进行操作的各种神经网络方法；
为操作结构化知识，产生结构化行为提供了 a straightforward interface.

+ 讨论了graph networks 如何支持relational reasoning和combinatorial generalization，为更复杂的、解释性更强的、更灵活的推理模式
奠定了基础。
## 1 introduction
+ combinatorial generalization

人类智能的一项关键能力—— **make infinite use of finite means**，即以无限的方式多产地组合有限的元素。这其实反映了**泛化组合的原则**：
从已知的building block中构建新的inferences、predictions、behaviors. 

人类的泛化组合能力依赖于 人类对关系的结构化表示和推理的认知机制：我们将复杂的系统表示为实体和实体间关系的组合；我们使用层级结构来
从fine-grained differences中提取more general commonalities between representations and behaviors；我们通过组合已有的技能和经验
来解决新的问题；我们通过对齐2个domains间的relational structure来进行类比，然后基于其中一个的知识来对另一个进行推断。

Kenneth Craik在《the nature of explanation》中，将世界的成分结构与人类的内在心理模式的组织方式联系在一起（以下内容并非原文翻译）：
> 人类在认知外部世界的relational structure时，大脑中已存在相似的relational structure来模拟这些外界的relational structure，
这些大脑中的“working physical model”和他们所代表的外界过程有着相同的工作方式。

人类通过一种组合的方式来理解世界。当学习新知识时，通常采取两种方法：
> 1. 将新知识加入到已有的结构化表示中；
> 2. 调整已有的structure，使其同时适用于新的和旧的知识。
+ structured approaches & modern deep learning methods

曾经结构化方法对机器学习十分重要，这是因为：数据难以获取，且计算代价十分昂贵，而通过结构化方法的强归纳偏置可以
提高样本复杂度，这十分有价值。

现代深度学习方法通常遵循端到端的设计哲学，这种思想强调了最小的先验表示和计算假设，试图避免显示结构和人工特征工程。
在数据获取变得极为容易，计算资源变得廉价时，这种模型十分灵活，并且能自己适应数据，于是使得为了能够更灵活的学习而
牺牲样本效率成为一种合理的选择。我们看到在机器翻译领域，在不使用语言实体间的显性解析树或者复杂关系的情况下，
sequence2sequence模型十分高效。
+ critiques about deep learning methods（深度学习方法面临的挑战）
> 1. 处理复杂的语言和场景理解
> 2. 推理结构化的数据
> 3. 训练条件之外的转移学习，以及从少量经验中学习

为应对这些挑战，研究者们做了一些十分有建设性的努力：在类比、语言分析、符号操作和其他形式的关系推理等领域，以及更多关于思维方式的综合理论中，
开发了各种创新的亚符号方法来表示和推理结构化对象。

参考文献：(Marcus, 2001; Shalev-Shwartz et al., 2017; Lake et al., 2017; Lake and Baroni, 2018; Marcus, 2018a,b; Pearl, 2018; Yuille and
Liu, 2018)  ☞6.1
+ previous graph-based reasoning

倡导一种综合方法来提高AI的combinatorial generalization，反对将灵活性和结构化对立起来，而是将二者结合用于建模。也就是说，
structured approaches和deep learning的手段要相互结合，二者相辅相成，才能取得更好的效果。

最近出现了一些将深度学习和structured approaches相结合的学习模型，具体来讲，其实就是处理graph数据的模型。这些模型
都有一个共同点：能够在离散的实体，以及这些离散实体间的关系上进行计算。他们与传统方法的区别在于：**实体和关系，以及其对应的计算的
表示和结构是如何被学习的？并且这些方法都带有很强的关系归纳偏置，以特定结构假设的形式指导这些方法学习实体和关系**。

参考文献：(e.g. Scarselli et al., 2009b; Bronstein et al., 2017; Gilmer et al., 2017; Wang et al., 2018c; Li
et al., 2018; Kipf et al., 2018; Gulcehre et al., 2018). ☞6.2
## 2 relational inductive biases

### 2.1 several basic concepts
+ relational reasoning关系推理

+ inductive biases归纳偏置
### 2.2 relational inductive biases in standard deep learning building blocks
+ fully connected layers

+ convolutional layers

+ recurrent layers
### 2.3 computations over sets and graphs
## 3 Graph Networks
### 3.1 Backgroud
### 3.2 Graph network block- main unit of computation in GNs
#### 3.2.1 definition of **Graph**
#### 3.2.2 internal structure of a GN block
#### 3.2.3 computational steps within a GN block\
#### 3.2.4 relational inductive biases in graph networks
## 4 Design principles for graph network architectures
### 4.1 flexible representations灵活的表示
### 4.2 configurable within-block structure可配置的块内结构
#### 4.2.1 为什么说block内部是可配置的？如何实现？
#### 4.2.2 几种不同的internal GN block配置
+ a full GN

+ an independent,recurrent block

+ an MPNN

+ a NLNN

+ a relational network

+ a Deep Set 
### 4.3 composable multi-block architectures可组合的多块结构
### 4.4 implementating graph networks in code
### 4.5 summary
## 5 discussion
## 5.1 combinational generalization in graph networks
## 5.2 limitations of graph networks
## 5.3 open questions
## 5.4 integrative approaches for learning and stucture
## 5.5 conclusion

## 6 references
### 6.1 critiques about deep learning methods
Marcus, 2001; Shalev-Shwartz et al., 2017; Lake et al., 2017; Lake and Baroni, 2018; Marcus, 2018a,b; Pearl, 2018; Yuille and
Liu, 2018
### 6.2 previous graph-based reasoning
(e.g. Scarselli et al., 2009b; Bronstein et al., 2017; Gilmer et al., 2017; Wang et al., 2018c; Li
et al., 2018; Kipf et al., 2018; Gulcehre et al., 2018)

