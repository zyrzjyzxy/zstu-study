import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def histogram_equalization(image):
    # 获取图像高度和宽度
    height, width = image.shape
    pixel_count = height * width
    
    # 计算直方图
    histogram = np.zeros(256)
    for i in range(height):
        for j in range(width):
            histogram[image[i,j]] += 1
    
    # 计算累积分布函数
    cdf = np.zeros(256)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
    
    # 归一化 CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (pixel_count - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    # 均衡化图像
    equalized_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            equalized_image[i,j] = cdf_normalized[image[i,j]]
    
    return equalized_image

def plot_histogram(image, title):
    plt.figure(figsize=(8, 4))
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)
    plt.xlabel('像素值')
    plt.ylabel('频率')

# 读取灰度图像
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 进行直方图均衡化
equalized_image = histogram_equalization(image)

# 显示原始图像和均衡化后的图像
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(equalized_image, cmap='gray')
plt.title('均衡化后的图像')
plt.axis('off')

# 显示直方图
plot_histogram(image, '原始图像直方图')
plot_histogram(equalized_image, '均衡化后的直方图')

# 保存结果
cv2.imwrite('original.jpg', image)
cv2.imwrite('equalized.jpg', equalized_image)

plt.show()