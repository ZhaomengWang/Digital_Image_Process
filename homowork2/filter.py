import cv2 as cv
import numpy as np
import math
import copy
import scipy.stats


#最大值滤波器
MAX_FILTER = 0
#最小值滤波器
MIN_FILTER = 1
#几何均值滤波器
GEOMETRIC_MEAN_FILTER = 2
#谐波均值滤波器
HMEANS_FILTER = 3
#逆谐波均值
IHMEANS_FILTER = 4
#中点滤波器
MINPOINT_FILTER = 5
#中值滤波器
MEDIAN_FILTER = 6
#修正的阿尔法均值滤波器
ALPHA_MEAN_FILTER_AMEND = 7
#自适应局部降低噪声滤波器
SELF_ADAPTING_FILTER = 8

'''*****************算术均值滤波器*****************'''
def spilt( a ):
    if a/2 == 0:
        x1 = x2 = a/2
    else:
        x1 = math.floor( a/2 )
        x2 = a - x1
    return -x1,x2

def original (i, j, k,a, b,img):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return  temp

def average_filter(a , b ,img):
    img0 = copy.copy(img)
    for i in range (0 , img.shape[0]  ):
        for j in range (2 ,img.shape[1] ):
            for k in range (img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i,j,k] = int ( np.mean(temp))
    return img

'''*****************几何均值滤波器*****************'''
def geometricMeanOperator(roi):
    '''
    计算当前模板的几何均值滤波器
    :param roi:
    :return:
    '''
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return np.power(p, 1/(roi.shape[0]*roi.shape[1]))

def geometricMeanSingle(img):
    '''
    几何均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    resultImg = np.copy(img)
    tempImg = cv.copyMakeBorder(img, 1,1,1,1,cv.BORDER_DEFAULT)
    for i in range(1, tempImg.shape[0]-1):
        for j in range(1, tempImg.shape[1]-1):
            resultImg[i-1, j-1] = geometricMeanOperator(tempImg[i-1:i+2, j-1:j+2])
    return resultImg.astype(np.uint8)

def img_gemotriccMean(img):
    '''
    几何均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return geometricMeanSingle(img)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = geometricMeanSingle(r)
        g = geometricMeanSingle(g)
        b = geometricMeanSingle(b)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

'''*****************谐波均值滤波器*****************'''
def HMeanOperator(roi):
    '''
    计算当前模板的算术均值滤波器
    :param roi:
    :return:
    '''
    roi = roi.astype(np.float64)
    if 0 in roi:
        return 0
    else:
        return scipy.stats.hmean(roi.reshape(-1))

def HMeanSingle(img, size = (3,3)):
    '''
    谐波均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    width = int(size[1]/2)
    height = int(size[0]/2)
    resultImg = np.copy(img)
    tempImg = cv.copyMakeBorder(img, height,height,width,width,cv.BORDER_DEFAULT)
    for i in range(height, tempImg.shape[0]-height):
        for j in range(width, tempImg.shape[1]-width):
            resultImg[i-height, j-width] = HMeanOperator(tempImg[i-height:i+height+1, j-width:j+width+1])
    return resultImg.astype(np.uint8)

def HMean(img, size=(3,3)):
    '''
    谐波均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param size: 滤波器大小
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return HMeanSingle(img, size)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = HMeanSingle(r, size)
        g = HMeanSingle(g, size)
        b = HMeanSingle(b, size)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))


'''*****************逆谐波均值滤波器*****************'''
def iHMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))

def iHMeanSingle(img, q):
    '''
    逆谐波均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param q: 指数
    :return:
    '''
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, 1,1,1,1,cv.BORDER_DEFAULT)
    for i in range(1, tempImg.shape[0]-1):
        for j in range(1, tempImg.shape[1]-1):
            temp = tempImg[i-1:i+2, j-1:j+2]

            resultImg[i - 1, j - 1] = iHMeanOperator(temp,q)

    resultImg = (resultImg - np.min(img))*(255/np.max(img))
    return resultImg.astype(np.uint8)

def iHMean(img, q):
    '''
    逆谐波均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param q: 指数
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return iHMeanSingle(img,q)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = iHMeanSingle(r, q)
        g = iHMeanSingle(g, q)
        b = iHMeanSingle(b, q)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))



'''*************************统计排序滤波器************************'''

'''*************************中值滤波器************************'''
def mid_filter(a, b, img):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, a, b, img0)
                img[i, j, k] = np.median(temp)
    return img



'''*************************最大值滤波器************************'''
def maxFilterOperator(roi):
    '''
    最大值滤波器，对模板内的部分进行计算
    :param roi: 部分图像矩阵
    :return:
    '''
    return np.max(roi)

def maxFilterSingle(img, size=(3,3)):
    '''
    最大值滤波器，单通道
    :param roi:
    :param size: rows,cols
    :return:
    '''
    width = int(size[1]/2)
    height = int(size[0]/2)
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, height,height,width,width,cv.BORDER_DEFAULT)
    for i in range(height, tempImg.shape[0]-height):
        for j in range(width, tempImg.shape[1]-width):
            resultImg[i-height, j-width] = maxFilterOperator(tempImg[i-height:i+height+1, j-width:j+width+1])
    return resultImg.astype(np.uint8)

def maxFilter(img, size=(3,3)):
    '''
    最大值滤波器
    :param img:
    :param size:
    :return:
    '''
    if size[0]%2 != 0 and size[1]%2 != 0and size[0] == 0 and size[1] == 0:
        raise (RuntimeError("滤波器大小必须为奇数"))
    dimension = len(img.shape)
    if dimension == 2:
        return maxFilterSingle(img,size)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = maxFilterSingle(r, size)
        g = maxFilterSingle(g, size)
        b = maxFilterSingle(b, size)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

'''*************************最小值滤波器************************'''
def minFilterOperator(roi):
    '''
    最小值滤波器，对模板内的部分进行计算
    :param roi: 部分图像矩阵
    :return:
    '''
    return np.min(roi)



'''*************************中点滤波器************************'''
def minpointFilterOperator(roi):
    return (int(np.max(roi))+int(np.min(roi)))/2


'''*************************修正的aloha均值滤波器************************'''
def alphaMeanFilterAmendOperator(roi, d):
    '''
    修正的aloha均值滤波器
    :param roi:
    :param d:
    :return:
    '''
    tempArray = roi.flatten()
    size = len(tempArray)
    if 2*d >= size:
        raise (RuntimeError("d不能超过滤波器元素的一半"))
    tempArray.sort()
    alphaAmendArray = tempArray[int(d/2):int(size-d/2)]
    return np.mean(alphaAmendArray)

'''*************************自适应局部降低噪声滤波器************************'''
def selfAdaptingFilter(roi, currentValue,sigma):
    '''
    自适应局部降低噪声滤波器
    :param roi:
    :param currentValue: 当前值
    :param sigma: 噪声方差
    :return:
    '''
    #局部均值
    m = np.mean(roi)
    #局部方差
    var = np.var(roi)
    return currentValue-(sigma/var)*(currentValue-m)

'''*************************自适应中值滤波器滤波器************************'''
def medianFilterOperator(roi):
    '''
    中值滤波器
    :param roi:
    :return:
    '''
    tempArray = roi.flatten()
    size = len(tempArray)
    tempArray.sort()
    return tempArray[int(size/2)]
def adaptive_median_filter_single(img,maxSize=3):
    '''
    自适应中值滤波器
    :param img:
    :return:
    '''
    maxLength = int(maxSize / 2)
    resultImg = np.zeros(img.shape)
#    print(resultImg.shape)
    tempImg = cv.copyMakeBorder(img, maxLength, maxLength, maxLength, maxLength, cv.BORDER_DEFAULT)
#    print(tempImg.shape)
    for i in range(maxLength, tempImg.shape[0] - maxLength):
        for j in range(maxLength, tempImg.shape[1] - maxLength):
            size = 3
            while size <= maxSize:
                length = int(size/2)
                roiImg = tempImg[i - length:i + length + 1, j - length:j + length + 1]
                zMin = int(np.min(roiImg))
                zMax = int(np.max(roiImg))
                zMed = int(medianFilterOperator(roiImg))
                zXY = int(tempImg[i,j])

                if zMed-zMin >0 and zMed-zMax < 0 :
                    if zXY - zMin >0 and zXY - zMax < 0:
                        resultImg[i - maxLength, j - maxLength] = zXY
                        break
                    else:
                        resultImg[i - maxLength, j - maxLength] = zMed
                        break
                else:
                    size = size + 2
                if size > maxSize :
                    resultImg[i - maxLength, j - maxLength] = zMed
    return resultImg.astype(np.uint8)

def adaptive_median_filter(img,maxSize=3):
    '''
    自适应中值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param maxSize: 滤波器最大大小
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return adaptive_median_filter_single(img,maxSize)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = adaptive_median_filter_single(r,maxSize)
        g = adaptive_median_filter_single(g,maxSize)
        b = adaptive_median_filter_single(b,maxSize)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

def imgFilter(img, size=3,filterType=MAX_FILTER, q=1, d=2,varOfNoise=0):
    '''
    滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param size: 滤波器大小，size=（rows,cols）
    :param filterType: 滤波器类型，在constants类中有定义
    :return:
    '''
    if size%2 != 0 and size == 0 :
        raise (RuntimeError("滤波器大小必须为奇数"))
    dimension = len(img.shape)
    if dimension == 2:
        return imgFilterSingle(img,size, filterType,q,d,varOfNoise)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = imgFilterSingle(r,size, filterType,q, d,varOfNoise)
        g = imgFilterSingle(g,size, filterType,q, d,varOfNoise)
        b = imgFilterSingle(b,size, filterType,q,d ,varOfNoise)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

def imgFilterSingle(img, size=3,filterType=MAX_FILTER, q=1,d=2,varOfNoise=0):
    '''
    :param img:
    :param size:
    :param filter:
    :return:
    '''
    #定义filterType和method之间的映射
    typeToMethod = {
        MAX_FILTER : maxFilterOperator,
        MIN_FILTER : minFilterOperator,
        GEOMETRIC_MEAN_FILTER : geometricMeanOperator,
        HMEANS_FILTER : HMeanOperator,
        IHMEANS_FILTER : iHMeanOperator,
        MINPOINT_FILTER: minpointFilterOperator,
        ALPHA_MEAN_FILTER_AMEND : alphaMeanFilterAmendOperator,
        SELF_ADAPTING_FILTER: selfAdaptingFilter
    }
    method = typeToMethod.get(filterType)
    length = int(size/2)
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, length,length,length,length,cv.BORDER_DEFAULT)
    for i in range(length, tempImg.shape[0]-length):
        for j in range(length, tempImg.shape[1]-length):
            if IHMEANS_FILTER == filterType:
                resultImg[i - length, j - length] = method(
                    tempImg[i - length:i + length + 1, j - length:j + length + 1],q)
            elif ALPHA_MEAN_FILTER_AMEND == filterType:
                resultImg[i - length, j - length] = method(
                    tempImg[i - length:i + length + 1, j - length:j + length + 1],d)
            elif SELF_ADAPTING_FILTER == filterType:
                resultImg[i - length, j - length] = method(
                    tempImg[i - length:i + length + 1, j - length:j + length + 1],tempImg[i,j],varOfNoise)
            else:
                resultImg[i-length, j-length] = method(tempImg[i-length:i+length+1, j-length:j+length+1])
    return resultImg.astype(np.uint8)


if __name__ == "__main__":

    #高斯噪声下的测试
    img0 = cv.imread(r"./gauss_noise.jpg")

    # 算术均值滤波器5*5滤波器大小
    ave_gauss_img = average_filter(5, 5, copy.copy(img0))
    cv.imwrite("./ave_gauss_img.jpg", ave_gauss_img)

    # 几何均值滤波器5*5滤波器大小
    geometric_gauss_img = imgFilter(img0, 5, GEOMETRIC_MEAN_FILTER)
    cv.imwrite("./geometric_gauss_img.jpg", geometric_gauss_img)

    # 谐波均值滤波器5*5滤波器大小
    hmeans_gauss_img = imgFilter(img0, 5, HMEANS_FILTER)
    cv.imwrite("./hmeans_gauss_img.jpg", hmeans_gauss_img)

    # 逆谐波均值滤波器5*5滤波器大小
    ihmeans_gauss_img = imgFilter(img0, 5, IHMEANS_FILTER)
    cv.imwrite("./ihmeans_gauss_img.jpg", ihmeans_gauss_img)

    # 最大值滤波器5*5滤波器大小
    max_gauss_img = imgFilter(img0, 5, MAX_FILTER)
    cv.imwrite("./max_gauss_img.jpg", max_gauss_img)

    # 最小值滤波器5*5滤波器大小
    min_gauss_img = imgFilter(img0, 5, MIN_FILTER)
    cv.imwrite("./min_gauss_img.jpg", min_gauss_img)

    # 中值滤波器5*5滤波器大小
    med_gauss_img = mid_filter(5, 5, copy.copy(img0))
    cv.imwrite("./med_gauss_img.jpg", med_gauss_img)

    # 中点滤波器5*5滤波器大小
    mid_gauss_img = imgFilter(img0, 5, MINPOINT_FILTER)
    cv.imwrite("./mid_gauss_img.jpg", mid_gauss_img)

    # 修正的阿尔法均值滤波器5*5滤波器大小
    alpha_gauss_img = imgFilter(img0, 5, ALPHA_MEAN_FILTER_AMEND)
    cv.imwrite("./alpha_gauss_img.jpg", alpha_gauss_img)

    # 自适应局部降低噪声滤波器5*5滤波器大小
    self_gauss_img = imgFilter(img0, 5, SELF_ADAPTING_FILTER)
    cv.imwrite("./self_gauss_img.jpg", self_gauss_img)

    #自适应中值滤波器滤波器5*5滤波器大小
    adaptive_gauss_img = adaptive_median_filter(img0,5)
    cv.imwrite("./adaptive_gauss_img.jpg", adaptive_gauss_img)