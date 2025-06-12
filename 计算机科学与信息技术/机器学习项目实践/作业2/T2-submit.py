import cv2
import numpy as np
import os


def find_license_plate_and_cluster(image_path, k=2):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None, None, None

    original_image_with_box = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    possible_plates = []
    min_area = 500
    max_area = 30000
    min_aspect_ratio = 2.0
    max_aspect_ratio = 5.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if aspect_ratio > min_aspect_ratio and aspect_ratio < max_aspect_ratio:
                rect_area = w * h
                extent = float(area) / rect_area
                if extent > 0.6:
                    possible_plates.append((x, y, w, h))

    best_plate_rect = None
    if possible_plates:
        possible_plates.sort(key=lambda p: p[2] * p[3], reverse=True)
        best_plate_rect = possible_plates[0]

    if best_plate_rect is None:
        print("未检测到车牌")
        return original_image_with_box, None, None

    x, y, w, h = best_plate_rect
    padding = 2
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)
    license_plate_roi = img[y_start:y_end, x_start:x_end]

    if license_plate_roi.size == 0:
        print("提取的车牌区域为空")
        return original_image_with_box, None, None

    cv2.rectangle(
        original_image_with_box, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2
    )

    pixel_values = license_plate_roi.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_plate = segmented_data.reshape(license_plate_roi.shape)
    color1 = centers[0]
    color2 = centers[1]
    is_blue_white = False
    if (color1[0] > 100 and color1[1] < 150 and color1[2] < 150) or (
        color2[0] > 100 and color2[1] < 150 and color2[2] < 150
    ):
        if (color1[0] > 150 and color1[1] > 150 and color1[2] > 150) or (
            color2[0] > 150 and color2[1] > 150 and color2[2] > 150
        ):
            is_blue_white = True
    return original_image_with_box, license_plate_roi, segmented_plate


if __name__ == "__main__":
    image_file = "T2.png"
    if not os.path.exists(image_file):
        print(f"文件 '{image_file}' 不存在")
    else:
        original_with_box, plate_roi, segmented_result = find_license_plate_and_cluster(
            image_file, k=2
        )
        if plate_roi is not None and segmented_result is not None:
            cv2.imshow("Original", original_with_box)
            cv2.imshow("Extracted", plate_roi)
            cv2.imshow("Segmented", segmented_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif original_with_box is not None:
            cv2.imshow("Original - Plate not found", original_with_box)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
