import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP
from dezero import Dataloader
import matplotlib.pyplot as plt
import time

max_epoch = 5
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

    # sum_loss, sum_acc = 0, 0
    # with dezero.no_grad():
    #     for x, t in test_loader:
    #         y = model(x)
    #         loss = F.softmax_cross_entropy_simple(y, t)
    #         acc = F.accuracy(y, t)
    #         sum_loss += float(loss.data) * len(t)
    #         sum_acc += float(acc.data) * len(t)
    # ## 存储loss和acc值
    # avg_test_loss = sum_loss / len(test_set)
    # avg_test_acc = sum_acc / len(test_set)
    # # test_metrics.append((avg_test_loss, avg_test_acc))
    # print('test_loss: {:.4f} test_acc: {:.4f}'.format(
    #     avg_test_loss, avg_test_acc))

# ## 解包
# train_losses, train_accuracies = zip(*train_metrics)  ## *将train_metrics中的每个元素解包成单独的参数传递给zip函数
# test_losses, test_accuracies = zip(*test_metrics)  ## train_metrics = [(0.5, 0.8), (0.4, 0.85), (0.35, 0.9)]
# ## 等效于：train_losses, train_accuracies = zip((0.5, 0.8), (0.4, 0.85), (0.35, 0.9))
# ## 之后zip函数再将这些元组的对应元素打包成新的元组，即将每个输入元组的第一个元素打包成一个元组，第二个元素打包成另一个元组
#
# ## 绘制曲线
# plt.subplot(121)  # 子图为1行两列，当前设置第1个子图，子图序号从左到右，由上至下来排序
# plt.plot(range(max_epoch), train_losses, color='red', label='train_loss')
# plt.plot(range(max_epoch), test_losses, color='blue', label='test_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.title('loss_results')
#
# plt.subplot(122)  # 子图为1行两列，当前设置第1个子图，子图序号从左到右，由上至下来排序
# plt.plot(range(max_epoch), train_accuracies, color='red', label='train_acc')
# plt.plot(range(max_epoch), test_accuracies, color='blue', label='test_acc')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(loc='lower right')
# plt.title('acc_results')
# plt.show()
