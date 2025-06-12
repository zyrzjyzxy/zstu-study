import cv2
import numpy as np
import os


def find_license_plate_and_cluster(image_path, k=2):
    """
    自动定位图像中的车牌，并使用K-means聚类分割车牌区域。

    Args:
        image_path (str): 输入图像的文件路径。
        k (int): K-means聚类的簇数 (默认为2)。

    Returns:
        tuple: 包含以下元素的元组:
            - original_image (numpy.ndarray): 带有定位框的原始图像。
            - license_plate_roi (numpy.ndarray): 提取出的车牌区域图像 (如果找到)。
            - segmented_plate (numpy.ndarray): K-means聚类分割后的车牌图像 (如果找到)。
            - None, None, None: 如果未找到车牌。
    """
    # --- 1. 读取图像 ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法加载图像: {image_path}")
        return None, None, None

    original_image_with_box = img.copy()  # 用于绘制最终结果

    # --- 2. 车牌定位 (基于颜色和轮廓) ---
    # 转换为HSV色彩空间，更容易分离颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义中国车牌蓝色的大致范围 (HSV)
    # 这些值可能需要根据实际光照和车牌情况微调
    # H: 100-130, S: 40-255, V: 40-255
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([130, 255, 255])

    # 创建蓝色的掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # (可选) 形态学操作去除小的噪点或连接断开的区域
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 去除小噪点
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # 填充小的空洞

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    possible_plates = []
    min_area = 500  # 车牌区域最小面积阈值
    max_area = 30000  # 车牌区域最大面积阈值
    min_aspect_ratio = 2.0  # 车牌最小宽高比
    max_aspect_ratio = 5.0  # 车牌最大宽高比

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # 检查宽高比是否在合理范围内
            if aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio:
                # 计算轮廓面积与边界矩形面积的比率，进一步筛选
                rect_area = w * h
                extent = float(area) / rect_area
                if extent > 0.6:  # 过滤掉过于空心的矩形
                    possible_plates.append((x, y, w, h))
                    # 在原始图上画出候选框（调试用，可选）
                    # cv2.rectangle(original_image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 假设最大的候选区域是车牌 (这是一种简化假设，可能需要更复杂的逻辑)
    best_plate_rect = None
    if possible_plates:
        # 可以根据面积、位置等选择最佳候选，这里简单选择第一个找到的或面积最大的
        # 按面积降序排序
        possible_plates.sort(key=lambda p: p[2] * p[3], reverse=True)
        best_plate_rect = possible_plates[0]  # 选择面积最大的

    if best_plate_rect is None:
        print("未检测到可能的车牌区域。")
        return original_image_with_box, None, None

    # --- 3. 提取车牌 ROI ---
    x, y, w, h = best_plate_rect
    # 可以稍微扩大一点边界，避免切到字符
    padding = 2
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)

    license_plate_roi = img[y_start:y_end, x_start:x_end]

    if license_plate_roi.size == 0:
        print("提取的车牌区域为空。")
        return original_image_with_box, None, None

    # 在原始图像上绘制最终定位框
    cv2.rectangle(
        original_image_with_box, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2
    )

    # --- 4. 对车牌区域进行 K-means 聚类分割 (K=2) ---
    # 将图像数据转换为 K-means 需要的格式 (n_samples, n_features)
    # n_samples 是像素总数, n_features 是 3 (BGR)
    pixel_values = license_plate_roi.reshape((-1, 3))
    # 转换为 float32 类型
    pixel_values = np.float32(pixel_values)

    # 定义 K-means 终止条件
    # (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 执行 K-means 聚类
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 将中心点颜色转换为 uint8
    centers = np.uint8(centers)

    # 将每个像素映射到其对应的中心点颜色
    segmented_data = centers[labels.flatten()]

    # 将数据重塑回原始图像的形状
    segmented_plate = segmented_data.reshape(license_plate_roi.shape)

    # 检查聚类结果的颜色，确保大致是蓝色和白色
    # 可以通过计算两个中心点的颜色来判断
    color1 = centers[0]
    color2 = centers[1]
    # 简单判断：哪个更像蓝色 (B分量大，R/G分量相对小)
    # 注意：这只是一个粗略的检查，实际颜色可能因光照变化很大
    is_blue_white = False
    if (color1[0] > 100 and color1[1] < 150 and color1[2] < 150) or (
        color2[0] > 100 and color2[1] < 150 and color2[2] < 150
    ):
        if (color1[0] > 150 and color1[1] > 150 and color1[2] > 150) or (
            color2[0] > 150 and color2[1] > 150 and color2[2] > 150
        ):
            is_blue_white = True

    print(f"聚类中心颜色 (BGR): {centers.tolist()}")
    if is_blue_white:
        print("聚类结果大致分离为蓝色和白色系。")
    else:
        print("警告：聚类结果的颜色可能不是预期的蓝色和白色。")

    return original_image_with_box, license_plate_roi, segmented_plate


# --- 主程序 ---
if __name__ == "__main__":
    image_file = "T2.png"  # 使用您上传的文件名

    if not os.path.exists(image_file):
        print(f"错误: 文件 '{image_file}' 不存在。请确保文件路径正确。")
    else:
        # 执行处理
        original_with_box, plate_roi, segmented_result = find_license_plate_and_cluster(
            image_file, k=2
        )

        # 显示结果
        if plate_roi is not None and segmented_result is not None:
            cv2.imshow("Original", original_with_box)
            cv2.imshow("Extracted", plate_roi)
            cv2.imshow("Segmented", segmented_result)

            print("\n按任意键关闭所有窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif original_with_box is not None:
            cv2.imshow("Original Image (Plate not found)", original_with_box)
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
