import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
plt.scatter(x, y, color='blue', label='Data points')
# plt.show()

w = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, w) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    w.cleargrad()
    b.cleargrad()

    loss.backward()
    w.data -= lr * w.grad.data
    b.data -= lr * b.grad.data
    print(w, b, loss)

x_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = predict(x_test)
# outputs = mean_squared_error(y, y_pred)
# y.name = 'y'
# y_pred.name = 'y_pred'
# plot_dot_graph(outputs, verbose=False, to_file='test.png')
plt.plot(x_test, y_pred.data, color='red')
plt.show()

