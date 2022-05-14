# softmax回归概述
## softmax回归是什么
>有时候我们的输出值不是一个值，而是一串离散的值（比如识别手写数字），这时候我们不能简单地线性地表示结果（线性的结果可以任意大，难以判断此时应该是哪一个离散的值，线性值与离散值之间的误差也难以衡量），这时候我们就需要将结果表示为概率，将所有值压缩在$(0,1]$间

这时候就需要所谓的逻辑函数(sigmod函数)，不同的是，softmax在输出结果等于2时退化回逻辑回归


**[原博客地址](https://blog.csdn.net/qq_43211132/article/details/102668037)**

[![softmax与logistic的区别](../pic/softmax%26logistic.png)](https://blog.csdn.net/qq_43211132/article/details/102668037)

softmax函数和之前的线性函数一样，是全连接层的，也就是说，它的每一个结点都与上一层的所有结点相连

[![SoftmaxNet](../pic/softmaxNet.svg)](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.4_softmax-regression)

## Softmax公式

对所有结果取指数相加作为整体，再用单个节点的指数除以这个整体就是该节点的概率
$$
\hat{y_{1}},\hat{y_{2}},\hat{y_{3}} = softmax(o_{1},o_{2},o_{2}),
$$
$$
\hat{y_{j}} = \frac{exp(o_{j})}{\sum_{i=1}^{3}exp(o_{i})} (j = 1,2,3)
$$

[![公式](../pic/softmax%E5%85%AC%E5%BC%8F.png)](https://blog.csdn.net/qq_43211132/article/details/102668037)

因此模型公式为：
$$
\hat{Y} = softmax(XW+b)
$$

## Softmax回归的Loss函数——交叉熵损失函数

因为平方差损失函数总是要求模拟结果和预期结果完全拟合，对Softmax回归这种多个输出计算概率的回归过于严格，因此应该采用一种只要某一节点的概率远高于其他节点就判定拟合的Loss函数，因此我们采用交叉熵损失函数：
对第i组数据：
$$
H(y^{(i)}, \hat{y}^{(i)}) = -\sum_{j=1}^{q} y_{j}^{(i)}\log {\hat{y}_{j}^{(i)}},其中j为模拟的第j个节点
$$
因此Loss函数是：

$$
l(\Theta) = \frac{1}{n} \sum_{i=1}^{n} H(y^{(i)}, \hat{y}^{(i)}) = \frac{1}{n} \sum_{i=1}^{n} y_{j}^{(i)}\log {\hat{y}_{j}^{(i)}},其中n为当前批次的数据个数
$$

又因为每个结果的真实值（概率）不是0就是1，且只有一个节点真实值为1，所以所谓的

$$
-\sum_{j=1}^{q} y_{j}^{(i)}\log {\hat{y}_{j}^{(i)}}
$$

其实只有一项，为

$$
\-y\_{k}^{(i)} \log {\hat{y}\_{k}^{(i)}}=\-\log {\hat{y}\_{k}^{(i)}}
$$

其中k为概率为1的节点k

因此交叉熵损失函数就是真节点（概率为1的节点）的模拟值的对数之和的均值的负数

同时我们也可以看到，Loss函数等价于：

$$
exp(nl(\Theta)) = \prod_{i=1}^{n} \hat{y}_{k}^{(i)}
$$

因此最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率

---
感谢[奔跑的小仙女](https://blog.csdn.net/qq_43211132?type=blog)的[博客](https://blog.csdn.net/qq_43211132/article/details/102668037)
