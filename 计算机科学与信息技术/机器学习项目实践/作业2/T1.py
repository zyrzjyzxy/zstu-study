import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 添加中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS 的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --- 参数 ---
image_path = "T1.png"  # 替换为你的图片路径
k = 5  # 聚类的数量

# --- 加载图像 ---
# 使用 OpenCV 加载图像 (默认 BGR 格式)
img = cv2.imread(image_path)

# 检查图像是否成功加载
if img is None:
    print(f"错误：无法加载图像: {image_path}")
else:
    print(f"图像加载成功，尺寸: {img.shape}")  # (height, width, channels)

    # --- 图像预处理 ---
    # 将图像从 BGR 转换为 RGB (可选，但 Matplotlib 显示时更直观)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像数据从 (height, width, channels) 变形为 (height * width, channels)
    # K-Means 需要一个二维数组，其中每一行是一个样本（像素），每一列是一个特征（颜色通道）
    pixel_values = img_rgb.reshape((-1, 3))

    # 将数据类型转换为 float32，K-Means 通常需要浮点数
    pixel_values = np.float32(pixel_values)

    # --- K-Means 聚类 ---
    # 定义 K-Means 模型
    # n_clusters: 指定聚类数量 (K)
    # n_init: 算法使用不同质心种子运行的次数。最终结果将是 inertia 最低的那个。
    #         设为 'auto' 或一个大于 1 的整数（如 10）来运行多次。
    # random_state: 为了结果可复现，可以设置一个随机种子
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=50, max_iter=300)

    # 拟合模型
    print("开始 K-Means 聚类...")
    kmeans.fit(pixel_values)
    print("聚类完成.")

    # 获取聚类中心（每个簇的平均颜色）和每个像素的标签（属于哪个簇）
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # --- 重建分割后的图像 ---
    # 将聚类中心（浮点数）转换回 uint8 整数类型
    centers = np.uint8(centers)

    # 根据每个像素的标签，用其对应的聚类中心颜色替换原始像素颜色
    segmented_image_flat = centers[labels.flatten()]

    # 将一维数组重新变形回原始图像的形状 (height, width, channels)
    segmented_image = segmented_image_flat.reshape(img_rgb.shape)

    # --- 显示结果 ---
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("原始图像")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f"分割后图像 (K={k})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --- (可选) 保存分割后的图像 ---
    # 需要将 RGB 转回 BGR 才能用 cv2.imwrite 正确保存
    # segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f'segmented_k{k}.png', segmented_image_bgr)
    # print(f"分割后的图像已保存为 segmented_k{k}.png")
