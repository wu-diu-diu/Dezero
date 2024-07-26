import numpy as np
from dezero import Variable
import dezero.functions  as F

x = Variable(np.random.rand(2, 3))
y = x.T
print(y)
