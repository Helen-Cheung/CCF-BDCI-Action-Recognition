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


## AGCN
AGCN论文中总结了ST-GCN模型的大致缺点: 其骨架图是根据人体关节连接预定义好的，无法修改 。
改进方案：提出了一种自适应的图神经网络，也就是引入了两个额外的参数化（可学习）的邻接矩阵，这两个矩阵分别用来（1）学习所有数据中的共同模式（也就是所有数据中统一的共性关注点）；（2）学习单个数据中独有的模式（也就是每个数据中独有的关注点）；

![image](https://user-images.githubusercontent.com/62683546/155316320-bdc25ff4-6afd-42ae-b6f4-e5018c73b7b3.png)

* 邻接矩阵由三个矩阵组成：Ak,Bk,Ck
* Ak为最基本的邻接矩阵
* Bk与STGCN中的Learnable edge importance weight相似,是一个参数化的NXN的矩阵,被设计来学习所有数据中的共同模式（也就是所有数据中统一的共性关注点）。
* Ck来学习单个数据中独有的模式（也就是每个数据中独有的关注点）,获取过程类似于视觉注意力中的Non-local
* 最后Ak、Bk、Ck按位相加获得邻接矩阵
* 之后的结构与ST-GCN基本一致

## MS-G3D
MS-G3D论文主要改进了GCN+TCN的时空处理结构，GCN+TCN范式中存在效率不高的时空信息流动方式。

![image](https://user-images.githubusercontent.com/62683546/155319752-e18d8590-6a57-48d0-8787-d4ce81b9c441.png)

为此，MS-G3D提出了MS-G3D模块，同步提取不同空间尺度、不同时间跨度的时空信息：

![image](https://user-images.githubusercontent.com/62683546/155318697-6a177262-b2c8-4940-abc0-d017f02a85cf.png)

通过将标准化的子邻接矩阵A重复了NXN次获得维度为tNxtN的新邻接矩阵，子邻接矩阵表示当前节点与距其距离1的t帧内所有节点的连接关系，其中t为时间窗大小，即每个时间窗包含t帧的骨骼数据。

网络的整体结构为：

![image](https://user-images.githubusercontent.com/62683546/155319940-5fd265be-7e26-4118-a123-027e253dc75a.png)

## PoseC3D
一种基于 3D-CNN 的骨骼动作识别方法

GCN 方法的 3 点缺陷：
* 鲁棒性： 输入的扰动容易对 GCN 造成较大影响，使其难以处理关键点缺失或训练测试时使用骨骼数据存在分布差异（例如出自不同姿态提取器）等情形。
* 兼容性： GCN 使用图序列表示骨架序列，这一表示很难与其他基于 3D-CNN 的模态（RGB, Flow 等）进行特征融合。
* 可扩展性：GCN 所需计算量随视频中人数线性增长，很难被用于群体动作识别等应用。

PoseC3D整体框架：

![image](https://user-images.githubusercontent.com/62683546/155320490-df376965-2f86-44db-a8f5-476f82b46869.png)


