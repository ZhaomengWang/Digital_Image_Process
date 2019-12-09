import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


# 在直线 y = 3 + 5x 附近生成高斯分布随机点
X = np.arange(0, 5, 0.1)
Z = [3 + 5 * x for x in X]
Y = [np.random.normal(z, 0.5) for z in Z]

plt.plot(X, Y, 'bo')
plt.savefig('./test.jpg')
img = Image.open("./test.jpg","r")
line_pixTuple_b = (0, 0, 255, 1)   #蓝色线

for n in range(10):
    i = random.randint(0,img.size[0]-10)
    y = random.randint(0,img.size[1]-10)
    for j in range(10):
        for k in range(10):
            img.putpixel((i + j, y + k), line_pixTuple_b)

img.save("./test.jpg")