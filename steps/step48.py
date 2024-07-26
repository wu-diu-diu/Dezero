import math
import numpy as np
import dezero
from dezero import optimizer
import dezero.functions as F
from dezero.models import MLP

# 超参数设置
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

## 数据集设置
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizer.SGD(lr).setup(model)
data_size = len(x)
max_iter = math.ceil(data_size / batch_size)  ## 小数点向上取整

## 训练
for epoch in range(max_epoch):
    ## 数据索引打乱
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]  ## 0:30, 30:60, 60:90
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)  ## loss.data是批次的平均损失，故需要乘以批次大小，得到每个批次的总损失，并求和

    avg_loss = sum_loss / data_size  ## 每轮训练后的样本总损失，除以样本数，得到样本的平均损失
    print('epoch: %d, loss: %.2f' % (epoch + 1, avg_loss))
