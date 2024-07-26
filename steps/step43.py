import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero import Parameter

# 数据集
np.random.seed(1)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
plt.scatter(x, y, color='blue', label='data point')
# plt.show()

# 权重初始化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 推理
def predict(x):
    y1 = F.matmul(x, W1) + b1
    y2 = F.sigmoid(y1)
    y3 = F.matmul(y2, W2) + b2
    return y3

lr = 0.2
iters = 10000

# 训练
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    W2.cleargrad()
    b1.cleargrad()
    b2.cleargrad()

    loss.backward()

    W1.data -= lr * W1.grad.data
    W2.data -= lr * W2.grad.data
    b1.data -= lr * b1.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(loss)

x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = predict(x_test)
plt.plot(x_test, y_pred.data, color='red', label='prediction')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test')
plt.show()
