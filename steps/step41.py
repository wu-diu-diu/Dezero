import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2, 3))
w = Variable(np.random.randn(3, 5))
y = F.matmul(x, w)
y.backward()

print(x.grad.shape)
print(w.grad.shape)