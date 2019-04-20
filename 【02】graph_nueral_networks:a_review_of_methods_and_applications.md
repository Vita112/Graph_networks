一篇关于图神经网络的综述，有以下几个贡献contributions in this paper:
> 1. provide a detailed review over existing graph neural network models. concretely, introduce the original model, its variants 
and several general frameworks. **provide a unified representations to present different propagation steps in different models**.
使用本文提出的representation，通过识别对应的aggregators和updaters，我们可以轻松地把不同的模型区分开来。
> 2. systematically categorize the applications and divide the applications into sturctural scenarios, non-structural scenarios and
other scenarios.
> 3. propose 4 open problems for future research.**graph neural networks suffer from over-smoothing and scaling problems**.在处理dynamic 
graphs，建模non-structural sensory data时，there is no effective methods.

**graph neural networks are connectionist models that capture the dependence of graphs via message passing between the nodes of graphs.
GNNs是图领域的深度学习方法，可以保持在 一个以任意深度表示其邻域信息 的状态**. 近期，**graph convolutional network(GCN) and gated graph neural network(GGNN)**在很多任务上， 
表现出突破性的进展。
