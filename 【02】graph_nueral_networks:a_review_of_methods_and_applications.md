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
graphs have the great expressive power of graphs:  can be used as denotation of a large number of systems across various areas including
social science, natural science, knowladge graphs and other research areas.

graphs analysis focuses on node classification, link prediction and clustering. 下面开始介绍图神经网络的几个基本动机：
+ the first motivation roots in convolutional neural networks

CNNs具有抽取多尺度局部空间特征，以及把它们组合成高度表达力的表示。**但是，CNNs只能处理 像图像的2D grid 和文本的1D sequence这类的 欧几里得数据，而这些
数据结构可以看作是 graphs 的 instances。CNNs的核心是：local connection局部连接, shared weights共享权重 and the use of multi-layer使用多层.**

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

focus on problems defined on graphs and we also investigate other mechanisms used in graph neural networks ,比如 gete mechanism、attention mechanism、skip connection。
## 2 Models
![notations_used_in_this_paper]()
## 2.1 Graph Neural Networks- original graph neural networks
在一个图中，每个node天然地由它的featurs和the related nodes定义。GNN的目标是：learn a state embedding $h_{v}\in \Re ^{s}$, 这个state embedding包含每个节点的邻居的信息。$h_{v}$是一个s-dimension vector of node v，被用于产生一个输出$o_{v}$,比如a node label。本文定义f为
一个parametric function - local transition function局部转移函数，这个函数在所有节点中共享，而且根据input neighborhood更新节点状态；定义g为local output function局部输出函数，描述输出如何产生。于是有：
$$h_{v}= f(X_{v},X_{co\[v]},h_{ne\[v]},X_{ne\[v]})\\\\
o_{v}=g(h_{v},X_{v})$$
其中，$X_{v},X_{co\[v]},h_{ne\[v]},X_{ne\[v]}$分别是v的feature，顶点edges的features，the states，features of the nodes in the neighbourhood of v。定义**H, O, X, $X_N$分别是 通过堆栈所有的状态， 所有的输出，所有的特征，所有的节点特征 而得到的向量，于是：
$$H=F(H,X)\\\\
O=G(H,X_{N})$$
其中，F是global transition function，G是global output function，分别由在一个图中堆栈所有节点的f和g得到。




