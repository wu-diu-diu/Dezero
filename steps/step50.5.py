import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP
from dezero import Dataloader


## 超参数设置
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

## 数据集设置
train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
## 迭代器设置
train_loader = Dataloader(train_set, batch_size)
test_loader = Dataloader(test_set, batch_size, shuffle=False)
## 模型设置
model = MLP((hidden_size, 3))
## 优化器设置
optimizer = optimizers.SGD(lr).setup(model)


for epoch in range(max_epoch):
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

    print('epoch: {:d} train_loss: {:.4f}, accuracy: {:.4f}'.format(
        epoch + 1, avg_train_loss, avg_train_acc))


