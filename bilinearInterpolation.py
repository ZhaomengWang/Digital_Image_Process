import numpy as np
import cv2
import math

def bi_linear(src, dst, target_size):
    pic = cv2.imread(src)       # 读取输入图像
    th, tw = target_size[0], target_size[1]
    emptyImage = np.zeros(target_size, np.uint8)
    for k in range(3):
        for i in range(th):
            for j in range(tw):
                # 首先找到在原图中对应的点的(X, Y)坐标
                corr_x = (i+0.5)/th*pic.shape[0]-0.5
                corr_y = (j+0.5)/tw*pic.shape[1]-0.5
                point1 = (math.floor(corr_x), math.floor(corr_y))   # 左上角的点
                point2 = (point1[0], point1[1]+1)
                point3 = (point1[0]+1, point1[1])
                point4 = (point1[0]+1, point1[1]+1)

                fr1 = (point2[1]-corr_y)*pic[point1[0], point1[1], k] + (corr_y-point1[1])*pic[point2[0], point2[1], k]
                fr2 = (point2[1]-corr_y)*pic[point3[0], point3[1], k] + (corr_y-point1[1])*pic[point4[0], point4[1], k]
                emptyImage[i, j, k] = (point3[0]-corr_x)*fr1 + (corr_x-point1[0])*fr2

    cv2.imwrite(dst, emptyImage)
    # 用 CV2 resize函数得到的缩放图像
    new_img = cv2.resize(pic, (200, 300))
    cv2.imwrite('pic/1_cv_img.png', new_img)

def main():
    src = './figure.png'
    dst = './new_figure.png'
    target_size = (300, 200, 3)     # 变换后的图像大小

    bi_linear(src, dst, target_size)

if __name__ == '__main__':
    main()