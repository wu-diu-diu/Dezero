import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP
from dezero import Dataloader
import matplotlib.pyplot as plt
import time

max_epoch = 40
batch_size = 100
hidden_size = 1000


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)
train_loader = Dataloader(train_set, batch_size)
test_loader = Dataloader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

# 创建列表以存储每个 epoch 的损失值和准确率
train_metrics = []
test_metrics = []

# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()
    test_loader.to_gpu()
    print("=====================================GPU Model===========================================")

for epoch in range(max_epoch):
    start = time.time()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    ## 存储loss和acc值
    avg_train_loss = sum_loss / len(train_set)
    avg_train_acc = sum_acc / len(train_set)
    # train_metrics.append((avg_train_loss, avg_train_acc))
    elapsed_time = time.time() - start

    print('epoch: {:d} train_loss: {:.4f}, accuracy: {:.4f}, time: {:.4f}[sec]'.format(
        epoch + 1, avg_train_loss, avg_train_acc, elapsed_time))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    ## 存储loss和acc值
    avg_test_loss = sum_loss / len(test_set)
    avg_test_acc = sum_acc / len(test_set)
    # test_metrics.append((avg_test_loss, avg_test_acc))
    print('test_loss: {:.4f} test_acc: {:.4f}'.format(
        avg_test_loss, avg_test_acc))
