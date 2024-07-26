from dezero.models import VGG16
import dezero
from PIL import Image
import numpy as np

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/cock.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
img.show()
x = VGG16.preprocess(img)
x = x[np.newaxis]  ## 在最前面添加一个轴1，添加用于处理小批量的轴

model = VGG16(pretrained=True)
with dezero.test_mode():  ## 不使用dropout
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])


