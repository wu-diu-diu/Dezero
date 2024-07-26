import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

# 数据集
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 超参数设置
lr = 0.2
iters = 10000
hidden_size = 10


# 定义模型
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)
losses = []

# 训练
for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    losses.append(loss.data)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model(x_test)
plt.subplot(121)  # 子图为1行两列，当前设置第1个子图，子图序号从左到右，由上至下来排序
plt.plot(x_test, y_pred.data, color='red', label='prediction')
plt.scatter(x, y, label='data point')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.title('prediction results')

# 图2
plt.subplot(122)  # 子图为1行两列，当前设置第二个子图
plt.plot(np.arange(iters), losses, label='loss curve')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.title('loss value')

plt.suptitle('subplot test')
plt.show()
