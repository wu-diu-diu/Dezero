import numpy as np
from dezero.models import MLP
import dezero.functions as F

x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
t = np.array([2, 0, 1, 0])

model = MLP((10, 5, 3))
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
