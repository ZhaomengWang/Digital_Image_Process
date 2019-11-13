from PIL import Image
import math

def generateFigure(imgW, imgH):

    img = Image.new("RGB", (imgW, imgH))  #创建一个指定大小的图片

    bg_pixTuple = (255, 251, 240, 1)  #三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度,在此处背景为乳白色
    line_pixTuple_r = (255, 0, 0, 1)  #红色线
    line_pixTuple_g = (0, 255, 0, 1)  #绿色线
    line_pixTuple_b = (0, 0, 255, 1)   #蓝色线

    for i in range(imgW):
        for j in range(imgH):
            img.putpixel((i, j), bg_pixTuple)

    w = (2 * math.pi) / imgW
    A = imgH/6

    #绘制正弦图像
    for i in range(imgW-10):
        y = int(A * math.sin(w * i) + A)
        for j in range(10):   #一个像素点太小了看不清，双重循环目的是加粗线条, 此处将线条宽度设为10像素，可自由修改
            for k in range(10):
                img.putpixel((i+j, y+k), line_pixTuple_r)

    #绘制余弦图像
    for i in range(imgW-10):
        y = int(A * math.cos(w * i) + 3*A) #向下平移了3个振幅，以便错开
        for j in range(10):
            for k in range(10):
                img.putpixel((i+j, y+k), line_pixTuple_g)

    #绘制余弦图像
    a = int(imgH / math.pow(w * (imgW-10), 2))
    for i in range(imgW - 10):
        y = imgH - 10 - int(a * math.pow((w * i), 2))
        for j in range(10):
            for k in range(10):
                img.putpixel((i + j, y + k), line_pixTuple_b)

    img.save("figure.png")


generateFigure(1024,768)
