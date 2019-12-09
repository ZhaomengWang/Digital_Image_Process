from skimage import io
import random
import numpy as np


def salt_and_pepper_noise(img, proportion=0.05):
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def gauss_noise(image):
    img = image.astype(np.int16)#此步是为了避免像素点小于0，大于255的情况
    mu =0
    sigma = 10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i,j,k] = img[i,j,k] +random.gauss(mu=mu,sigma=sigma)
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    return img

def rayleigh_noise(img,scale=1):
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
        rayNoise = np.random.rayleigh(scale, size=(height, width))
    elif dimension == 3:
        height, width, channel = img.shape
        rayNoise = np.random.normal(scale, size=(height, width, channel))
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

    noisy = img + rayNoise
    return noisy.astype(np.uint8)

if __name__ == '__main__':
    img = io.imread(r"./satomi.jpg")
    noise_img1 = salt_and_pepper_noise(img)
    io.imshow(noise_img1)
    io.imsave('./salt_and_pepper_noise.jpg', noise_img1)


    noise_img2 = gauss_noise(img)
    io.imshow(noise_img2)
    io.imsave('./gauss_noise.jpg', noise_img2)


