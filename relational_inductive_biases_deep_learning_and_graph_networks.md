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

我们将structure定义为 组合一组已知构建块的产物。“structured representations”捕获这个组成（即元素的排列），“structured computations”作为一个整体对元素及其组成进行操作。关系推理涉及操纵实体和关系的结构化表示，使用
关于如何组成他们的规则，部分术语解释如下：
> 1. 实体是具有属性的元素，例如具有大小和质量的物理对象；
> 2. 关系是实体之间的属性，两个物体之间的关系可能包括相同的尺寸、重量等；关系本身也可以具有属性，超过X倍的关系取一个属性X，它决定了关系的相对权重阈值是真是假；关系也可能对全局环境敏感，对于一个石头和一根羽毛，以更大的加速度下降这种关系依赖于上下文是在空气中，还是在真空中。**此处我们关注实体间的配对关系**。
> 3. 规则是一个函数（如非二进制逻辑谓词），它将实体和关系映射到其他实体和关系。

（Pearl，1988; Koller and Friedman，2009）演示了一个 在机器学习中应用图模型进行关系推理的实例，图模型可以通过在
随机变量之间进行显式随机条件独立来表示复杂的联合分布。**这些模型十分成功，因为①能够捕获 许多现实世界
生成过程的稀疏结构；②支持用于学习和推理的高效算法**。比如在隐马尔可夫模型中，在给定前一时间步状态的情况下下，限制latent states与其他状态条件独立，并且，在给定当前时间步latent state的情况下，限制观察值是条件独立的，这与许多现实世界的因果过程的关系结构十分匹配。

明确表达变量之间的稀疏依赖关系提供了各种有效的推理算法，比如message-passing，它们在图模型内部的各个地方之间应用通用的消息传递过程，从而产生 一个可组合的、部分可并行的推理过程，此过程可应用于不同尺寸和形状的图模型。
+ inductive biases归纳偏置

学习涉及搜索一个解决方案的空间，以期提供更好的数据解释，或者获得更高的回报。但在很多情况下，存在多种解决方案同样出色。**归纳偏置允许一种解决方案优先于另一种解决方案，并独立于观察数据；它对学习过程中实体间的关系和相互作用施加约束**。

在贝叶斯模型中，通常通过先验分布的选择和参数化来表达归纳偏置；归纳偏置也可以是一个正则化项，被编码在算法本身的构架中，来避免过拟合。**归纳偏置通常会以牺牲灵活性为代价，来提高样本的复杂性，并且可以根据偏差-方差权衡来理解**。理想的归纳偏置，既可以改善对解决方案的搜索，又不会明显降低性能，并帮助找到以理想方式推广的方法。

归纳偏置可以表达关于数据生成过程或解决方案空间的假设。举了2个例子：1. 当使用1-d函数来拟合数据时，线性最小二乘法遵循了一个限制：逼近函数应该是一个线性模型，并且在一个quadratic penalty二次惩罚下，approximation errors近似误差应该是最小的。**这反映出一个假设：数据生成过程可以被简单地解释为 一个被加性高斯噪声破坏的线性过程。**2。L2正则化优先考虑prioritize那些参数值更小的解决方案，并且可以针对其他不适定的问题ill-posed problems引入独特的解决方案和全局结构。**这可以被解释为一个关于学习过程的假设：当解决方案之间的模糊程度更小时，更容易找到好的解决方案**。注意：这些假设不需要是显式的，他们反映了模型或算法如何与世界交互。
+ elementary building blocks within creative new machine learning architectures in recent years

在创新的新的机器学习构架中，实践者们通常**遵循组合elemantary building blocks的设计模式，以形成更复杂，更深的计算层级结构和图形**。例如fully connected layers，convolutional layers其实都可以看作是building blocks，**这些layers的组合提供了一种特殊的relational inductive bias类型，即分层处理hierarchical processing，处理过程中计算是分阶段进行的，通常导致输入信号中信息间的长距离交互**。下图显示了：building blocks本身带有各种关系归纳偏置：

![relational_inductive_bias_in_standard_deep_learning_components](https://github.com/Vita112/Graph_networks/blob/master/img/relational_inductive_bias_in_standard_deep_learning_components.jpg)

除了上述这些关系归纳偏置外，深度学习中也使用了各种**非关系归纳偏置**，比如:激活非线性，权值衰减，丢弃法，批量和层归一化，数据增强data augmentation，training curricula训练课程和优化算法，这些都对学习轨迹和学习输出施加了限制。为方便探索在不同深度学习方法中的关系归纳偏置，理解不同构架间的 实体，关系以及规则的不同，探测每种构架如何支持关系推理，事先定义一下内容：
> 1. 规则函数的参数，例如哪些实体和关系作为输入；
> 2. 规则函数如何在计算图中复用或共享，比如跨不同实体和关系，跨不同时间或处理步骤；
> 3. 框架如何定义表示之间的 interaction和isolation，比如通过应用规则来得出关于相关实体的结论，而不是单独分别处理他们。

### 2.2 relational inductive biases in standard deep learning building blocks
+ fully connected layers全连接层
通过一个非线性向量值函数实现，input为vector，output vector的每个元素通过一个带有偏置向的 权重-向量点积操作得到。在全连接层中，**实体：units in the network；关系：all-to-all；规则：specified by the weights and biases；没有复用，没有信息隔离；隐式关系归纳偏置非常弱：所有输入单元相互作用以确定任何输出单元的值，并且在各输出间独立**。
+ convolutional layers

通过将input vector或者tensor与同等级的卷积核进行卷积，添加偏置项并应用逐点非线性来实现。**实体：individual units,or grid elements;但是关系更加稀疏**。在卷积层中有一些**重要的关系归纳偏置：局部性和平移不变性locality and
translation invariance**。局部性指的是 关系规则的参数是在输入信号坐标空间中彼此靠近的实体，与远端实体隔离；平移不变性指的是 输入中跨局部区域复用相同的规则。
+ recurrent layers

通过一系列steps来实现，**实体：每个处理步骤中的inputs和hidden states；关系：前一隐藏状态和当前输入的隐藏状态的马尔科夫依存；规则：将每一步的输入和隐藏状态作为参数，以更新隐层状态；规则在每个步骤中复用，反映出时间不变性temporal invariance的关系归纳偏置**。

![reuse_and_sharing_in_common_deep_learning_building_blocks](https://github.com/Vita112/Graph_networks/blob/master/img/reuse_and_sharing_in_common_deep_learning_building_blocks.jpg)
### 2.3 computations over sets and graphs
我们需要具有实体和关系的明确表示的模型，以及用于计算其交互的规则的学习算法，以及将它们置于数据中的方法。**注意：世界上的实体通常没有自然秩序，但是，可以通过实体间关系的属性进行排序**。通过关系推理的深度学习组件，应该反映这种顺序不变性，即用于处理sets和graphs的深度学习模型应该能够在不同排序方式下都有相同的结果。

集合是用于 由其顺序是不确定的或者不相关的实体描述的 系统的自然表示，**集合中实体间的关系归纳偏置不是relation的存在，而是顺序的缺席**。
> 我的理解：是否可以理解为 集合中的关系归纳偏置是由内部各元素的排序来决定的，而顺序可以通过实体间关系的属性确定，那么，集合中实体间关系的属性的定义应该是十分关键的？

为解释这句话☝，论文给出了第一个任务：预测n个行星组成的太阳系的质心。任务中，行星集合的顺序无关紧要，因为状态仅依据聚集的平均数量便可以被描述。**这个任务不可以使用MLP来解决，这是因为：在MLP中，针对一个特殊的input{x1,x2,……,xn}所学习到的预测并不一定会转化到一个 拥有不同顺序的同样的input中,即MLP对每一个每一个输入都是按照单独的方式处理，而不是一视同仁地处理所有输入。因为集合中存在n!个这样的可能置换，并且在更坏的情况下， 使用MLP将导致combinatorial explosion**。这个例子说明了处理sets时，应该遵循置换不变性permutation invariance的关系归纳偏置。

在许多问题中，置换不变性并不是唯一重要的基本结构形式。例如，**集合中的每个对象都可能受到与集合中其他对象的成对交互的影响**。现在考虑这样一个任务：预测行星系统中，在一个时间间隔δt之后的每个个体行星的位置。此时使用上段中的那种方法（使用聚合的平均信息）显然是不够准确的，因为每个行星的运动依赖于其他行星对其施加的力。在这个任务中，需要考虑系统的全局置换不变性，并且考虑了2个参数（这个例子中指的是2个行星）。

上述2个任务说明了2种关系结构：一种是实体间不存在任何关系；一种是每两个实体间存在配对关系。许多现实世界的系统拥有 
在这两个极端之间的某处的一个关系结构，就是说现实世界中，一些实体对拥有一个关系，而另一些则没有。
> 我的理解：哲学上我们讲，整个世界处在相互联系的体系中，任何事物不可能孤立于其他事物而单独存在，世界的本质是联系的，发展的，联系是普遍的，发展是曲折的。而认知世界需要这样的联系发展思维，如果能够捕获到现实世界中实体间的关系，他们是交互的，还是条件独立的，这将有助于我们理解更为复杂的现实世界。

回到太阳系的例子中，如果整个系统由行星及其各自的卫星组成，那么在预测时，我们可以通过忽略行星们各自的卫星之间的相互作用，来近似得到各个行星间的关系。**这种思想正好对应于图，因为在图体系中，第i个对象仅与 其邻域描述的其他对象的子集产生交互。注意，更新后的状态仍然不依赖于我们描述邻域的顺序**。

通常，图是一种 能够支持任意关系结构的表示，并且在图上的计算能够提供一个 超出卷积层和循环层能提供的 更强的关系归纳偏置。
## 3 Graph Networks
### 3.1 Backgroud
图神经网络对于 被认为具有丰富关系结构的任务中表现优秀，并在各种领域中得到了应用与发展（以下所列仅为部分）：
> 1. visual scene understanding tasks（Raposo et al.，2017; Santoro et al.，2017）
> 2. sew-shot learning（Garcia and Bruna，2018）
> 3. learn the dynamics of physical systems（Battaglia et al.，2016; Chang et al.，2017; Watters et al.，2017; van Steenkiste et al.，2018; Sanchez-Gonzalez et al.，2018） ）
> 4. reason about knowledge graphs（Bordes et al.，2013; On ~oro-Rubio et al.，2017; Hamaguchi et al.，2017）
> 5. perform semi-supervised text classification（Kipf and Welling，2017）
> 6. machine translation（Vaswani et al.，2017; Shaw et al.，2018; Gulcehre et al.，2018）
> 7. building generative models of graphs（Li et al.，2018; De Cao and Kipf，2018; You et al.，2018; Bojchevski et al.，2018）
> 8. unsupervised learning of graph embeddings（Perozzi et al.，2014; Tang et al.，2015; Grover and Leskovec，2016;Garc'ıa-Dura'n and Niepert，2017）

+ **一些应该引起注意的现有方法和评论**

Bronstein et al. (2017)提供了在非欧几里得数据上的 关于深度学习的优秀调研，探索了graph neural nets，graph convolution networks，related spectral approaches；

Gilmer et al. (2017)引入了消息传得神经网络MPNN，统一了各种图神经网络和图卷积网络的方法(Monti et al., 2017; Bruna et al., 2014; Hena· et al., 2015; De·errard et al., 2016;Niepert et al., 2016; Kipf and Welling, 2017; Bronstein et al., 2017)

Wang等人（2018c）引入了非局部神经网络（NLNN），它通过类比计算机视觉和图模型的方法，统一了各种“自我关注”式方法self-attention-style methods（Vaswani et al.，2017; Hoshen，2017; Velickovi'c et al.，2018），以捕获信号中的长距离依赖性。
# 接下来开始，进入本文的核心部分！！！
### 3.2 Graph network block- main unit of computation in GNs
本文中提出的GN框架为图形结构表示 定义了一类关系推理的函数，概括和扩展了各种图神经网络，MPNN,NLNN方法，支持从简单的构建块中，构建复杂的体系结构。

**但关于如何构建这个具体操作问题，本文似乎并没有叙述**。

GN框架的主要计算单元是GN block，它是一个graph-to-graph的模块，输入和输出都是graph，在结构上执行计算。GN 框架的block organization强调了可定制性，并且可以合成 表达关系归纳偏置的新结构。其设计原则将在后文中介绍。**为更具体地
理解GN的形成机制，考虑在一个任意重力场中预测一组橡胶球的运动，这些橡胶球不是相互弹跳，而是每一个都一个，或多个弹簧将它们连接到其他一些（或者全部）的弹簧上**。下图描绘了一些常见场景：

![different_graph_representations](https://github.com/Vita112/Graph_networks/blob/master/img/different_graph_representations.jpg)
#### 3.2.1 definition of **Graph**-什么是“图”？
图被定义为三元组G=(u, V, E),实体由图的nodes表示，关系由图的edges表示，global attributes表示system-level properties，结合下图：

![definition_of_graph_in_this_paper](https://github.com/Vita112/Graph_networks/blob/master/img/definition_of_graph_in_this_paper.jpg)

u：global attribute。比如表示引力场；

V：V={vi},nodes的集合，其中i=1:Nv,每一个vi是一个节点的属性。比如V代表每个球，具有位置，速度和质量的属性。

E：E={(ek, rk, sk)},edges的集合，其中k=1:Ne，每个ek表示边的属性，rk是接收节点的索引，rs是发送节点的索引。比如E表示不同球之间是存在弹簧的，而且他们对应的弹簧是固定的。
#### 3.2.2 internal structure of a GN block-一个GN block内部包含哪些组件？
一个GN block包含3个更新函数φ，和3个聚集函数ρ：

![internal_structure_of_a_GN_block](https://github.com/Vita112/Graph_networks/blob/master/img/internal_structure_of_a_GN_block.jpg)

φe被映射到所有edges以计算每个edge更新，φv被映射到所有nodes以计算每个node更新，并且φu被应用一次作为全局更新。每个ρ函数都将一个集合作为输入，并将其减少为表示聚合信息的单个元素。**ρ函数必须对其输入的排列不变，并且应该采用可变数量的参数（例如，元素求和，平均值，最大值等）***.
#### 3.2.3 computational steps within a GN block-一个GN block内部如何进行计算？
先看算法过程：

![algorithm_steps_of_computation_in_a_full_GN_block](https://github.com/Vita112/Graph_networks/blob/master/img/algorithm_steps_of_computation_in_a_full_GN_block.jpg)

下面对每一步进行解释：
> 1. 每个边应用φe，使用参数（ek，vrk，vsk，u），并返回e'k。$E_{i}^{'}$表示为每个节点i产生的每个边的输出的集合：
$$E_{i}^{'}={(e_{k}^{'},r_{k},s_{k})}\_{r_{k}=i,k=1:N^{e}}$$
弹簧示例中，对应于两个连接球之间的力或势能。
> 2. ρe→v应用于$E_{i}^{'}$，并将投影到顶点i的边缘更新聚合到$\bar{e}\_{i}^{'}$中，这将用于下一步的节点更新.弹簧示例中，对应于 对作用在第i个球上的所有力或者势能进行求和。
> 3. φv应用于每个节点i，以计算更新的节点属性$v_{i}^{'}$, 产出的每个节点输出的集合是：
$$V^{'}={v_{i}^{'}}\_{i=1:N^{v}}$$
弹簧示例中，对应于 计算更新每个球的位置、速度和动能等属性信息。
> 4. ρe→u应用于$E^{'}$，并将所有边缘更新聚合成$\bar{e}^{'}$，然后将用于下一步的全局更新。弹簧示例中，对应于 计算力的总和，和弹簧的势能。
> 5. ρv→u应用于$V^{'}$，并将所有节点更新聚合到$\bar{v}^{'}$，然后将用于下一步的全局更新.弹簧示例中，计算系统的总动能。
> 6. 每个图形应用φu一次，并计算全局属性$u^{'}$的更新.弹簧示例中，计算出与物理系统的静力和总能量类似的东西。

下图描述了 在边更新、节点更新、全局更新时，分别调用了哪些图元素：

![updates_in_a_GN_block]()

**注意：虽然我们给出了步骤顺序，但，不一定严格按照这个顺序执行**。比如，可以反转更新函数，以从全局，每个节点再到每个边的顺序更新。
#### 3.2.4 relational inductive biases in graph networks
1. graph can express arbitrary relations among entities,这意味着 **是GN的输入决定了representations是如何交互和被隔离的，而不是由固定的体系结构来决定这些选择**。具体的，实体的对应节点之间存在edge，则实体间的关系由edge表示；反之，不存在edge代表节点之间没有关系，不直接相互影响。

2. graphs represent entities and their relations as sets，which are invariant to permutations.这意味着 **GNs对集合中的元素的顺序具有不变性**。

3. a GN's per-edge and per-node functions are reused across all edges and nodes, respectively.（GN的每条边和每个节点函数分别在所有边和所有节点上复用），这意味着 **GNs自动支持一种组合泛化形式（参见5.1，即GNs的结构不仅严格地执行计算at system level，而且跨实体和跨关系地应用了shared computations），这允许对前所未见的系统进行推理），因为：graphs由edges，nodes以及global features组成，单个的GN可以在不同大小（由edges和nodes的数量决定）和形状（edge连接）上进行操作。
## 4 Design principles for graph network architectures
GN框架可以用于实现各种构架设计。**but，如何实现？**

通常，框架对于具体的属性表示和函数形式是不可知的。*此处，本文主要聚焦在深度学习构架上，这允许GNs充当可学习的graph-to-graph function approximators*。
### 4.1 flexible representations灵活的表示
通过2种方式，GNs support highly flexible graph representations
+ 1. in terms of the representations of the attributes依据属性表示

一个GN block的global、node和edge的属性可以使用任意的表示形式。比如深度学习中最常用real-valued vectors和tensors；也可以
使用诸如sequences、sets、甚至是graphs等其他的数据结构。**问题的要求通常决定了为属性使用哪种表示**，当input data is an image，属性一般被表示为tensors of image patches；当input data is a text document，属性可能被表示为对应句子的单词的序列。

在更广泛框架内的每个GN block，edge 和 node输出通常对应于 vectors和tensors 的列表，每个edge和node，以及global outputs对应于a single vector or tensor。**这使得GN的输出可以传给其他深度学习构建块，比如MLPs，CNNs以及RNNs。

a GN block的输出可以根据任务的需求进行定制：
> - an edge-focused GN边缘聚焦GN使用edges作为输出，比如对实体间的交互做决策。
> - an node-focused GN节点聚焦GN使用nodes作为输出，比如关于物理系统的推理。
> - an graph-focsed GN图聚焦GN使用全局变量作为输出，比如回答有关视觉场景的问题等。

nodes，edges，以及global outputs也可以根据任务进行混合和匹配。
+ 2. in terms of the structure of the graph itself依据图结构本身
当定义输入数据如何被表示为一个图时，通常存在2种情景：①输入明确规定了关系结构；②关系结构必须被推导出，或者被假设。

具有更明确规定的实体和关系的例子：knowledge graphs，social networks，parse tree，optimization problems等。

关系结构不明确，必须被推断，或假设的例子：visual scenes，text corpora，programming language source code，multi-agent system等，在这些类型的设置中，数据可能被格式化为一个没有关系的实体的集合，或仅仅只是一个vector或者tensor。

*如实体未明确规定，则可以假设它们，例如，通过 将一个句子的每个单词，或一个CNN输出特征映射中的每个局部特征向量 视为一个节点 (Watters et al., 2017; Santoro et al., 2017;Wang et al., 2018c) (Figures 2e-f)，或者使用一个单独的学习机制来推断来自非结构化信号的实体（Luong et al.，2015; Mnih et al.，2014; Eslami et al.，2016; van Steenkiste et al.，2018）。如果关系未被提供，最简单的方法是实例化实体间的所有可能有向边，但是，当实体很多时无法使用，因为可能的边缘数量将随着节点数量呈现二次方增长*。

**开发更复杂的方法来推断非结构化数据的稀疏结构，将是未来的重要方向**。
### 4.2 configurable within-block structure可配置的块内结构
再次给出等式1:

![internal_structure_of_a_GN_block](https://github.com/Vita112/Graph_networks/blob/master/img/internal_structure_of_a_GN_block.jpg)

上图中，每一个φ必须用函数f来实现，f的参数决定了 使用何种信息作为input。下图显示了不同内部配置的GN block：

![different_internal_GN_block_configurations](https://github.com/Vita112/Graph_networks/blob/master/img/different_internal_GN_block_configurations.jpg)

GN框架中，各种其他的构架可以作为不同的函数选择和块内配置。
#### 4.2.1 为什么说block内部是可配置的？如何实现？
#### 4.2.2 几种不同的internal GN block配置
+ a full GN

Hamrick等(2018)和Sanchez-Gonzalez等(2018)使用图4a中所示的full GN block，其中φ实现使用神经网络（在下面表示为NNe，NNv和NNu，表示它们是具有不同参数的不同函数）。他们的ρ实现使用元素和，但也可以使用平均和最大或最小，

![a_full_GN_block](https://github.com/Vita112/Graph_networks/blob/master/img/a_full_GN_block.png)

其中\[x，y，z]表示向量或张量拼接。for vector attributes，φ通常使用MLP；for tensors such as image feature maps，φ通常使用CNNs。
+ MPNN消息传递系统
> - 消息函数Mt，相当于GN中的φe，但是不接收u；
> - 用于GN的ρe→v的元素求和；
> - 更新函数Ut，相当于GN中的φv；
> - 读出函数R，相当于GN中的φu，但是不接收u或者E'，因此不需要与GN的ρe→u类似；
> - dmaster与GN的u大致相似，但是被定义为连接到所有其他节点的额外节点，因此不会直接影响edge和global updates，然后它可以在GN的V中表示。

+ NLNN非局部神经网络
Wang等（2018c）的NLNN统一了各种“intra-/self-/vertex-/graph-attention内/自/顶点/图注意”方法（Lin et al.，2017; Vaswani et al.，2017; Hoshen，2017; Velickovi'c et al.，2018; Shaw et al.，2018），可转换成GN形式。

attention指代 节点是如何更新的：每个节点的更新基于其邻居的 节点属性加权和，其中节点与其邻居之间的权重由它们属性之间的
标量成对函数计算得到（然后在邻居之间标准化）。已发布的NLNN计算所有节点之间的成对注意力加权。下图显示NLNN的结构：

![NLNNs_as_GNs](https://github.com/Vita112/Graph_networks/blob/master/img/NLNNs_as_GNs.png)

φe被分解为标量成对交互函数，返回两项：非标准化注意项$\alpha ^{e}(v_{r_{k}},v_{s_{k}}) = a_{k}^{'}$ 和 
向量值非成对项$\beta ^{e}(v_{s_{k}})=b_{k}^{'}$. 在ρe→v聚合中,$a_{k}^{'}$ 项在每个接收器的边缘上进行归一化，$b_{k}^{'}$ 和元素求和,计算公式如下：

![formula_of_NLNNs](https://github.com/Vita112/Graph_networks/blob/master/img/formula_of_NLNNs.png)

该公式可能有助于仅关注与下游任务最相关的那些交互，特别是当输入实体是一组时，通过在它们之间添加所有可能的边来形成图。
+ CommNet(Sukhbaatar et al.，2016), structure2vec(Dai et al.，2016), gated graph sequence neural networks门控图序列神经网络(Li et al.， 2016)

已使用不直接计算成对交互的φe，而是忽略接收节点，仅在发送方节点上操作，在某些情况下仅操作边缘属性。这可以通过具有以下签名的φe的实现来表达:

![ignore_the_reciever_node](https://github.com/Vita112/Graph_networks/blob/master/img/ignore_the_reciever_node.png)
+ relational network (Raposo et al., 2017; Santoro et al., 2017) 

忽视节点更新，直接从池化的边缘信息种预测全局输出：

![ignore_nodes_update](https://github.com/Vita112/Graph_networks/blob/master/img/ignore_nodes_update%5D.png)
+ a Deep Sets(Zaheer et al., 2017) 

完全忽视边缘更新，直接从池化的节点信息中，预测全局输出：

![ignore_edges_update](https://github.com/Vita112/Graph_networks/blob/master/img/ignore_edges_update.png)
### 4.3 composable multi-block architectures可组合的多块结构
图网络的一个关键设计原则是通过组合GN块来构建复杂的体系结构。我们定义了一个GN块，因为它始终将包含边，节点和全局元素的图作为输入，并返回一个与输出具有相同组成元素的图（当这些元素未明确更新时，只需将输入元素传递给输出）。这种图形到图形的输入/输出接口确保一个GN块的输出可以作为输入传递给另一个，即使它们的内部配置不同，类似于标准深度学习工具包的张量到张量接口。在最基本的形式中，两个GN块GN1和GN2可以通过将第一个输出作为输入,传递给第二个来组成GN1◦GN2,即G’ = GN2（GN1（G））。


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



## 7 some blogs 
1. [读书笔记7：Relational inductive biases, deep learning, and graph networks](https://blog.csdn.net/b224618/article/details/81380567)
