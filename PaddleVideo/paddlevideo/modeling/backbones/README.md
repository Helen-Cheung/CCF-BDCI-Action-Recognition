## ST-GCN
主要特点是：自动学习了data的Spatial（空间:同一时间节点与节点的连接关系）和temporal（时间：不同时间同一节点的连接关系）的特性。

在ST-GCN这篇文章中，作者的另一大创新点是通过对运动的分析引入了图划分策略，即建立多个反应不同运动状态（如静止，离心运动和向心运动）的邻接矩阵。作者在原文中提到其采用了三种不同的策略，分别为：

* Uni-labeling，即与跟根节点相邻的所有结点具有相同的label，如下图b所示。
* Distance partitioning，即根节点本身的label设为0，其邻接点设置为1，如下图c所示。
* Spatial configuration partitioning，是本文提出的图划分策略。也就是以根节点与重心的距离为基准（label=0），在所有邻接节点到重心距离中，小于基准值的视为向节心点（label=1），大于基准值的视为离心节点（label=2）。

![image](https://user-images.githubusercontent.com/62683546/155313976-cd1c68ba-8a61-4ac8-b11d-e3973c09405c.png)、

具体的代码实现如下：

```
A = []
for hop in valid_hop:
    a_root = np.zeros((self.num_node, self.num_node))
    a_close = np.zeros((self.num_node, self.num_node))
    a_further = np.zeros((self.num_node, self.num_node))
    for i in range(self.num_node):
        for j in range(self.num_node):
            if self.hop_dis[j, i] == hop:
                if self.hop_dis[j, self.center] == self.hop_dis[
                        i, self.center]:
                    a_root[j, i] = normalize_adjacency[j, i]
                elif self.hop_dis[j, self.
                                  center] > self.hop_dis[i, self.
                                                         center]:
                    a_close[j, i] = normalize_adjacency[j, i]
                else:
                    a_further[j, i] = normalize_adjacency[j, i]
    if hop == 0:
        A.append(a_root)
    else:
        A.append(a_root + a_close)
        A.append(a_further)
A = np.stack(A)
```

值得注意的是，hop类似于CNN中的kernel size。hop=0就是根节点自身，hop=1表示根节点与其距离等于1的邻接点们，也就是上图（a）的红色虚线框。

![11111](https://user-images.githubusercontent.com/62683546/155315224-f82ae065-f4ab-4cc0-b39e-d79372f3ce24.png)

可以分为以下步骤：


* 步骤1：引入一个可学习的权重矩阵（与邻接矩阵等大小）与邻接矩阵按位相乘。该权重矩阵叫做“Learnable edge importance weight”，用来赋予邻接矩阵中重要边（节点）较大的权重且抑制非重要边（节点）的权重。
* 步骤2：将加权后的邻接矩阵A与输入X送至GCN中进行运算，实现空间维度信息的聚合。
* 步骤3：利用TCN网络（实际上是一种普通的CNN，在时间维度的kernel size>1）实现时间维度信息的聚合。
* 步骤4：引入了残差结构（一个CNN+BN）计算获得Res，与TCN的输出按位相加

