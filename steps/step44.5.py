import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L


class TwoLayerNet(Model):  # 子类会继承父类所有的属性和方法，除非子类覆盖了这些方法
    def __init__(self, hidden_size, out_size):  # 子类中构造__init__方法会覆盖父类的init方法，故需在子类的init方法中先显示调用父类
        super().__init__()  # 的init方法
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


x = Variable(np.random.randn(5, 10), name='x')
model = TwoLayerNet(100, 10)
model.plot(x)  # TwoLayerNet继承了Model的plot方法，model是Two的实例，故其调用plot时，Model的plot代码在执行时的self是model
# 也即Two实例，故Model中的self.forward调用的是Two的forward方法。
# 此为python的动态绑定（或动态方法调用）的概念。Python 会在运行时根据对象的实际类型来决定调用哪个方法
