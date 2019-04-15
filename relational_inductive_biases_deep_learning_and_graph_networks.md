本文由DeepMind，Google Brain，MIT等机构的27位作者于2018年联合发表，文章的一大部分是对之前工作的整理和回顾，并提出了
graph-to-graph的Graph Networks，which intend to incorporate deep learning methods and relational reasoning,
and is hopeful to solve the problem of relational reasoning in deep learning. 以下开始是论文读书笔记：
## 0 abstract
## 1 introduction
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
### about

