import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
from dezero import optimizer

# 数据集
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 超参数设置
lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizer.MomentumSGD()
optimizer.setup(model)


for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)
