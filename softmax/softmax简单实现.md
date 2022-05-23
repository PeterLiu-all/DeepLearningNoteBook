# softmax简单实现

## 需要导入的库
```python
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
```

## 定义需要的函数

> 一次统计学习由学习的模型，学习的策略，学习的算法组成

### 定义模型函数

定义softmax函数，将非线性引入线性回归之中
```python
# 定义softmax


def softmax(Oi: torch.Tensor):
    Oi_exp = Oi.exp()
    return Oi_exp/(Oi_exp.sum(dim=1, keepdim=True))

# 定义模型


def net(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    return softmax(torch.mm(X.view((-1, W.size()[0])), W)+b)
```

### 定义交叉熵损失函数

形式上是对数损失函数

依靠经验误差对模型进行选择

```python
# 定义Loss函数
# 交叉熵损失函数
# 表现为对数损失函数


def Loss(y: torch.Tensor, y_hat: torch.Tensor):
    # 这里的y是n*1的矩阵，而y_hat是n*j
    return - torch.log(y_hat.gather(1, y.view(-1, 1))).sum()
```

### 定义准确率

```python
def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    # 如果某一次模型预测结果与真实结果相同，就返回1，否则返回0
    # 用argmax函数获取y_hat中最大项的索引（0~9），与真实结果对比
    # 转化为float后加和，取其中的值返回
    return (y_hat.argmax(1) == y).float().sum().item()
```

### 定义优化算法

和前面一样的

```python
# 定义优化算法


def sgd(params, lr, batch_size):
    for param in params:
        # 学习率就是一个用于控制下降幅度的常数
        # 权重和偏移优化
        param.data -= lr*param.grad / batch_size
```

### 定义数据分批的函数

```python
def data_iter(mnist_test, batch_size):
    return torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=0)
```

### 训练函数

只要获取到了训练好的权重和偏移，就相当于获得了训练好的模型

```python
def train(n_epoches: int, lr: float, w_hat: torch.Tensor, b_hat: torch.Tensor,
          batch: int, data: torch.Tensor) -> torch.Tensor:
    n_features = len(data)
    # 一共拟合训练数据n_epoch次
    for epoch in range(n_epoches):
        # 初始化某一次的经验误差，准确率
        train_l, ac = 0.0, 0.0
        # 随机梯度下降法，将数据分批
        for f, lb in data_iter(data, batch):
            y_hat = net(f, w_hat, b_hat)
            # 计算当前批次经验损失并与上一批的损失加和
            l = Loss(lb, y_hat).sum()
            train_l += l
            # 计算准确率
            ac += accuracy(lb, y_hat)
            # 反向求偏导
            l.backward()
            # 反向优化模型
            sgd([w_hat, b_hat], lr, batch)
            w_hat.grad.data.zero_()
            b_hat.grad.data.zero_()
        # 将误差总和，准确率总和除以训练样本容量
        train_l /= n_features
        ac /= n_features
        print(f"epoch {epoch}, loss:{train_l}, accuracy:{ac}")
    print(f"hat weight:{w_hat}")
    print(f"hat bias:{b_hat}")
    return w_hat, b_hat
```

### 测试训练成果

```python
def test(test_iter: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    ac = 0.0
    batch_size = len(test_iter)
    for f, lb in data_iter(test_iter, batch_size):
        y_hypo = net(f, w, b)
        ac += accuracy(lb, y_hypo)
    ac /= batch_size
    print(f"test accuracy:{ac}")
```

## 开始训练

```python
def main():
    batch_size = 256
    # 得到的是训练组和测试组
    # 用训练组来训练模型，用测试组来测试模型准确度
    train_iter = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                   transform=transforms.ToTensor())
    test_iter = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                                  transform=transforms.ToTensor())
    # 行数*列数，是一张图的像素总量
    num_inputs = 28*28
    # 标注问题，只有0~9一共10个标注
    num_outputs = 10

    # 初始化权重与偏移
    w = torch.tensor(np.random.normal(
        0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    # 训练模型
    w, b = train(10, 0.03, w, b, batch_size, train_iter)
    # 测试模型
    test(test_iter, w, b)


if __name__ == "__main__":
    main()

```

## 训练结果
```Plain
epoch 0, loss:1.0356025695800781, accuracy:0.69585
epoch 1, loss:0.709172248840332, accuracy:0.7767
epoch 2, loss:0.6368897557258606, accuracy:0.7977333333333333
epoch 3, loss:0.597575306892395, accuracy:0.80825
epoch 4, loss:0.5720041394233704, accuracy:0.8157666666666666
epoch 5, loss:0.5530234575271606, accuracy:0.8211666666666667
epoch 6, loss:0.5385252833366394, accuracy:0.8246333333333333
epoch 7, loss:0.5267878770828247, accuracy:0.8280666666666666
epoch 8, loss:0.5172721743583679, accuracy:0.8301
epoch 9, loss:0.5093177556991577, accuracy:0.8327666666666667
hat weight:tensor([[ 3.6310e-03, -7.5329e-03,  3.7053e-03,  ...,  3.1249e-02,
         -2.1517e-03,  2.2466e-03],
        [ 2.1203e-02, -8.7482e-03,  1.5493e-02,  ..., -2.7951e-03,
         -6.2426e-03,  2.5957e-02],
        [ 1.7465e-02,  1.5385e-03, -2.2081e-03,  ...,  1.4364e-02,
          2.8634e-03, -2.0297e-02],
        ...,
        [ 9.7792e-05,  3.9122e-03,  2.1818e-02,  ..., -1.7938e-02,
          2.0017e-02, -7.0280e-03],
        [-2.0205e-04, -1.5446e-02,  3.1180e-03,  ..., -1.0438e-02,
         -2.1133e-03, -6.5218e-03],
        [-3.9553e-03,  4.7967e-03, -1.0672e-02,  ..., -4.2892e-03,
         -4.3317e-04, -1.1061e-02]], requires_grad=True)
hat bias:tensor([ 0.0898, -0.0810, -0.0846,  0.0271, -0.4667,  1.0832,  0.2318, -0.0801,       
        -0.2585, -0.4609], requires_grad=True)
test accuracy:0.8204
```