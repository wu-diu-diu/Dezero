import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000
losses = []

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    losses.append(loss.data)

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

# 图1
x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = predict(x_test)
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


