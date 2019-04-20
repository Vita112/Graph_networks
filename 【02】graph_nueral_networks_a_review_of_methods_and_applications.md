一篇关于图神经网络的综述，有以下几个贡献contributions in this paper:
> 1. provide a detailed review over existing graph neural network models. concretely, introduce the original model, its variants 
and several general frameworks. **provide a unified representations to present different propagation steps in different models**.
使用本文提出的representation，通过识别对应的aggregators和updaters，我们可以轻松地把不同的模型区分开来。
> 2. systematically categorize the applications and divide the applications into sturctural scenarios, non-structural scenarios and
other scenarios.
> 3. propose 4 open problems for future research.**graph neural networks suffer from over-smoothing and scaling problems**.在处理dynamic 
graphs，建模non-structural sensory data时，there is no effective methods.

**graph neural networks are connectionist models that capture the dependence of graphs via message passing between the nodes of graphs.
GNNs是图领域的深度学习方法，可以保持在 一个以任意深度表示其邻域信息 的状态**. 近期，**graph convolutional network(GCN) and gated graph neural network(GGNN)** 在很多任务上，表现出突破性的进展。
## 0 introduction
graphs have the great expressive power of graphs:  can be used as denotation of a large number of systems across various areas including social science, natural science, knowladge graphs and other research areas.

graphs analysis focuses on node classification, link prediction and clustering. 下面开始介绍图神经网络的几个基本动机：
+ the first motivation roots in convolutional neural networks

CNNs具有抽取多尺度局部空间特征，以及把它们组合成高度表达力的表示。**但是，CNNs只能处理 像图像的2D grid 和文本的1D sequence这类的 欧几里得数据，而这些数据结构可以看作是 graphs 的 instances。CNNs的核心是：local connection局部连接, shared weights共享权重 and the use of multi-layer使用多层.**

上述CNNs的核心对 解决图领域的问题同样十分重要, 因为：
> 1. graphs are the most typical locally connected structure；
> 2. 与传统的光谱图理论相比，共享权重减少计算代价；
> 3. multi-layer 结构是解决hierarchical pattern层级结构的关键，可以捕获不同大小的特征。

+ other motivation comes from graph embedding

graph embedding learns to represent nodes, edges or subgraphs in low-dimentional vectors.**基于表示学习的第一个graph embedding method 是 DeepWalk，它在生成的random walks随机游走上应用了SkipGram model。但是，方法有2个严重的缺陷**：
> 1. 在encoder中，节点之间没有参数共享，意味着 参数的数量醉着节点的数量线性增长；
> 2. the direct embedding methods lack the ability of generalization,意味着 graphs 不能处理dynamic graphs， 或者生成新的图。

基于CNNs和graph embeding，GNNs可以对 由元素及其依赖性组成的输入和输出建模；同时，使用RNN kernel对图上的扩散过程进行建模。
+ why graph neural networks are worth investigating？

1. CNNs 和 RNNs按特定的顺序对节点特征进行堆栈叠加，因此它们无法处理graph input。在GNNs中，网络分别在每一个节点传播，忽略了the input order of nodes,也就是说** the output of GNNs is invariant for the input order of nodes**.

2. an edge in a graph represents the information of dependency between 2 nodes. **GNNs can do propagation guided by the graph sturcture instead of using it as part of nodes by a weighted sum of the states of their neighborhood(通过他们的邻居状态的加权和 来更新隐藏状态)**。

3. 推理是高水平AI的重要研究主题，而人类大脑的reasoning process主要基于来自日常经验的图。GNNs explore to generate the graph from non-structural data like images or documents.
## 1 related work
+ original graph nueral  networks 《The graph neural network model》in 2009

give a formal definition of early graph nueral network approaches.
+ a unified framework-MoNet 《geometric deep learning on graphs and manifolds using mixture model cnns》in 2017**非欧几里得数据:图和流形**

generalize CNN architectures to non-Euclidean domains(graph and manifolds) and the framework could generalize several spectral mothods on graphs or manifolds.

+ in this paper

focus on problems defined on graphs and we also investigate other mechanisms used in graph neural networks ,比如 gate mechanism、attention mechanism、skip connection。
## 2 Models
![notations_used_in_this_paper](https://github.com/Vita112/Graph_networks/blob/master/img/notations_used_in%E3%80%9002%E3%80%91.png)
### 2.1 Graph Neural Networks- original graph neural networks
+ 算法描述

在一个图中，每个node天然地由它的featurs和the related nodes定义; input graph由带标签信息的节点和undirected edges组成。GNN的目标是：learn a state embedding $h_{v}\in \Re ^{s}$, 这个state embedding包含每个节点的邻居的信息。$h_{v}$是一个s-dimension vector of node v，被用于产生一个输出$o_{v}$,比如a node label。本文定义f为一个parametric function - local transition function局部转移函数，这个函数在所有节点中共享，而且根据input neighborhood更新节点状态；定义g为local output function局部输出函数，描述输出如何产生。于是有：
$$h_{v}= f(X_{v},X_{co\[v]},h_{ne\[v]},X_{ne\[v]})\cdots (1)\\\\
o_{v}=g(h_{v},X_{v})\cdots (2)$$
其中，$X_{v},X_{co\[v]},h_{ne\[v]},X_{ne\[v]}$分别是v的feature，顶点edges的features，the states，features of the nodes in the neighbourhood of v。定义**H, O, X, $X_N$分别是 通过堆栈所有的状态， 所有的输出，所有的特征，所有的节点特征 而得到的向量**，于是：
$$H=F(H,X)\cdots (3)\\\\
O=G(H,X_{N})\cdots (4)$$
其中，F是global transition function，G是global output function，分别由在一个图中堆栈所有节点的f和g得到。
**接下来，计算state**：
$$H^{t+1}=F(H^{t},X)\cdots (5)$$
上述公式是一个dynamic system，并以指数级速度快速收敛得到$H=F(H,X)$的解。
+ **how to learn the parameters of f and g**?

使用监督学习方法，于是有target information $t_v$代表一个特定的节点，loss 可描述如下：
$$loss = \sum_{i=1}^{p}(t_{i}-o_{i})\cdots (6)$$
其中p是可监督节点的数量，使用梯度下降算法优化，有以下步骤：
> 1. 状态$h_{v}^{t}$通过公式1进行迭代得到，一直到时间T为止；
> 2. 梯度的权重W 通过loss，即公式6 计算；
> 3. 权重 W 根据最后一步计算出的gradient 进行更新

+ **limitations**

1. 对于fixed point，迭代更新节点的隐藏状态的效率低下。

2. GNN在迭代中使用相同的参数；节点隐藏状态的更新是一个序列过程，可以从 类似GRU,LSTM的RNN kernel中 获益。

3. 仍有一些informative features on the edges，which cannot be effectively modeled in the original GNN。比如，知识图谱中的edges拥有关系类型，并且，通过不同的edges的消息传播应该根据他们的类型的不同而不同。

4. 如果我们聚焦在nodes的表示，而不是graphs的表示，那么使用fixed points是不合适的。因为 在fixed point中的表示分布 会有更加平滑的值，而且，区分每个节点的有用信息会很少。
### 2.2 variants of graph neural networks
### 2.2.1 graph types - 扩展了原始模型的表示能力(representation capabilities of original model)
+ directed graphs有向图-《rethinking knowledge graph propagation for zero-shot learning》2018

*在原始模型中，输入图是由标签信息的节点和无向边构成的*。无向边可以看作是 2个directed edges，表示2个实体间存在着一个关系。然而，directed edges可以带来 比undirected edges更多的信息。比如在知识图谱中，边从head entity开始，在end entity结束，head entity是tail entity的parent class，这告诉我们：**我们应该区别对待 从parent classes到child classes和 从child classes到parent classes 的信息传递过程**。在此是有公式的，but，我看不懂！！提到了一个概念：[归一化邻接矩阵normalized adjacency matrix]()
+ heterogeneous graph异构图-《deep collective classification in heterogeneous information networks》2018

在异构图中，有好几种不同的nodes。处理异构图的最简单的方法是：把每个节点类型转化为一个 由节点本身的原始特征拼接而成的独热特征向量。**GraphInception将metapath的概念引入了异构图上的propagation，使用metapath，我们可以根据邻居的节点类型和到邻居的距离，来分组这些邻居。对每一个邻居组来说，GraphInception将它看作一个同构图homogeneous graph中的sub-graph来进行传播，并且拼接来自不同的homogeneous graphs的propagation results来得到一个聚集的节点表示**。

+ Graphs with edge information 
- 2 papers:G2S from《graph-to-sequence learning using geted graph neural networks》2018; r-GCN from 《modeling relational data with graph convolutional networks》2018

在图的最终变体中，每个edge都有边自身的信息，比如边的weight和type。处理这种图的2种方法：
> 1. 将图转化为一个二部图(convert a graph to a bipartite graph),于是，原始边变成了节点，而且，1个原始边被分成了2个新的边，这意味着 **在edge node和begin/end nodes之间，存在2个新的边**。**G2S的encoder使用下面这个aggregation function for neighbors**：
$$h_{v}^{t}=\rho (\frac{1}{|N_{v}|})\sum_{u\in N_{v}}\mathbf{W}\_{r}(r_{v}^{t}\odot h_{u}^{t-1}+b_{r})$$
其中，$\mathbf{W}\{r}$和$b_{r}$是不同的边(关系)类型的propagation parameters。

> 2. 在不同种类的edges上，应用不同的weight matrices for propagation。当关系数量很大时，r-GCN介绍了2种正则化方法来减少参数数量：bais-diagonal-decomposition基对角分解 和 block-diagonal-composition块对角分解。在基对角分解中，每一个$\mathbf{W}\{r}$被定义如下：
$$\mathbf{W}\_{r}=\sum_{1}^{B}a_{r_{b}}V_{b}$$
公式表明：$\mathbf{W}\{r}$是一个关于基变换$V_{b}\in \Re ^{d_{in}\times d_{out}}$的线性组合，只有系数$a_{r_{b}}$依赖于r。<br>
在块对角分解中，r-GCN通过 在一个低维矩阵集上直接求和 来定义每一个$\mathbf{W}\{r}$。
### 2.2.2 propagation types-several modifications on propagation step, and learn representations with higher qualilty
模型中的propagation step和output step对于获得 hidden state of nodes(or edges)是十分重要的。下图中列出了 对原始图神经网络的传播步骤进行一些主要修正后的GNNs的变体：

![an_overview_of_variants_of_graph_neural_networks](https://github.com/Vita112/Graph_networks/blob/master/img/an_overview_of_variants_of_graph_neural_networks.png)

接下来是GNN不同变体的比较：

![different_variants_of_graph_neural_networks](https://github.com/Vita112/Graph_networks/blob/master/img/different_variants_of_graph_neural_networks.png)

**不同的变体使用了不同的aggregators来聚集来自每一个节点邻居的信息，而且，使用特定的updaters来更新节点的隐藏状态**。
+ **1. convolution**

+ **2. gate**

+ **3. attention**

+ **4. Skip connection**

### 2.2.3 training methods
### 2.3 general frameworks通用框架
+ **1. message passing nueral networks**

+ **2. non-local neural networks**

+ **3. graph networks**

## 3 applications 
### 3.1 structural scenarios结构化场景
### 3.2 non-structural scenarios非结构化场景
### 3.3 other scenarios
## 4 open problems
### 4.1 shallow structure
### 4.2 dynamic graphs
### 4.3 non-structural scenarios
## 5 conclusion







