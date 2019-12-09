import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as sl
import pylab
from PIL import Image, ImageEnhance



def least_square_method(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转化为灰度图像
    frame = cv2.GaussianBlur(frame, (3,3), 0)#高斯模糊，去噪
    _, Imask = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)#二值化
    Imask = cv2.erode(Imask, None, iterations=2)
    Imask = cv2.dilate(Imask, np.ones((5, 5), np.uint8), iterations=2)
    sobely=cv2.Sobel(Imask,cv2.CV_64F,0,1,ksize=5)
    sobely=abs(sobely)
    _,c=cv2.threshold(sobely, 500, 255, cv2.THRESH_BINARY)#二值化
    row,col=c.shape
    c=c[int(row/5):int(row*4/5),int(col/3):int(2*col/3)]
    row,col=c.shape
    cv2.imshow('o',c)
    x=np.linspace(0,col,col)
    y=np.array(x)
    for i in range(col):
        cy=row-np.argmax(c[:,i])
        y[i]=cy
    xmean=x.mean()
    ymean=y.mean()
    k=((x*y).mean()-xmean*ymean)/(pow(x,2).mean()-(xmean**2))
    b=ymean-k*xmean
    X = np.arange(0, 5, 0.1)
    Y = [b + k * x for x in X]
    plt.plot(X, Y, 'k',linewidth=5)
    plt.show()



def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    '''
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    '''
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if len(also_inliers > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入

        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def ransac_test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 100, 1e4, 7, 30, debug=debug, return_all=True)

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图

    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit',linewidth=5)
    pylab.legend()
    pylab.show()

def img_processing(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # canny边缘检测
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    return edges

def hough_transform(img):
    img = ImageEnhance.Contrast(img).enhance(3)
    img = np.array(img)
    result = img_processing(img)
    # 霍夫线检测
    lines = cv2.HoughLinesP(result, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)
    print("Line Num : ", len(lines))

    # 画出检测的线段
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        pass
    img = Image.fromarray(img, 'RGB')
    img.save("hough_detect.jpg")

if __name__ == "__main__":
   img = cv2.imread('./real_pic.jpg',1)
   # least_square_method(img)
   #ransac_test()
   img = Image.open("./real_pic.jpg", "r")
   hough_transform(img)
