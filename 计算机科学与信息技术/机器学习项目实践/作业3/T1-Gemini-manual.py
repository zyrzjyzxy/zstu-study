# 导入所需的库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 仍然使用它来方便地计算准确率
import matplotlib.pyplot as plt
from collections import Counter  # 用于找出列表中最常见的元素 (众数)
import math  # 用于开平方根

# 添加中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS 的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# --- 1. 加载鸢尾花数据集 ---
iris = load_iris()
X = iris.data  # 特征
y = iris.target  # 目标标签

print(f"数据集包含 {X.shape[0]} 个样本和 {X.shape[1]} 个特征。")
print(f"目标类别: {iris.target_names}")
print("-" * 30)

# --- 2. 划分训练集和测试集 ---
# 使用与之前相同的参数，以确保可比性
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")
print("-" * 30)

# --- 3. 手动实现 kNN 相关函数 ---


# 定义计算欧氏距离的函数
def euclidean_distance(point1, point2):
    """计算两个数值型向量之间的欧氏距离"""
    distance = 0.0
    # 确保维度相同
    if len(point1) != len(point2):
        raise ValueError("输入点的维度必须相同")
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)
    # 或者使用 NumPy (通常更快):
    # return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


# 定义 kNN 预测函数 (针对单个测试点)
def knn_predict_single(X_train, y_train, test_point, k):
    """
    对单个测试点使用 kNN 进行预测。
    Args:
        X_train: 训练数据的特征。
        y_train: 训练数据的标签。
        test_point: 需要预测的单个测试点的特征。
        k: 邻居的数量。
    Returns:
        预测的类别标签。
    """
    distances = []
    # 1. 计算测试点到所有训练点的距离
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        # 存储 (训练标签, 距离) 对
        distances.append((y_train[i], dist))

    # 2. 根据距离对所有邻居进行排序 (升序)
    distances.sort(key=lambda x: x[1])

    # 3. 选取最近的 k 个邻居
    neighbors_labels = []
    # 确保 k 不超过训练样本总数
    k_actual = min(k, len(distances))
    if k_actual <= 0:
        # 如果 k 无效或没有训练数据，返回一个默认值或引发错误
        # 这里简单返回 -1 表示错误或无法预测
        return -1

    for i in range(k_actual):
        neighbors_labels.append(distances[i][0])  # 获取邻居的标签

    # 4. 找出 k 个邻居中最常见的类别标签 (投票法)
    # 使用 Counter 统计每个标签出现的次数，most_common(1) 返回出现次数最多的元素及其次数
    most_common = Counter(neighbors_labels).most_common(1)

    if not most_common:
        # 如果 neighbors_labels 为空 (例如 k=0 或异常情况)，返回默认值
        return -1

    return most_common[0][0]  # 返回最常见的标签


# --- 4. 寻找最优 K 值 (使用手动实现的 kNN) ---
k_range = range(1, 26)  # 与之前相同的 K 值范围
manual_accuracy_scores = []  # 存储手动实现的准确率

print("开始使用手动实现的 kNN 测试不同的 K 值...")
# 循环遍历 K 值范围
for k in k_range:
    y_pred_manual = []  # 存储对测试集的预测结果
    # 对测试集中的每一个点进行预测
    for test_point in X_test:
        prediction = knn_predict_single(X_train, y_train, test_point, k)
        y_pred_manual.append(prediction)

    # 计算当前 K 值下的整体准确率
    # 注意：这里仍然使用 accuracy_score 来比较预测结果和真实标签，以简化评估过程
    accuracy = accuracy_score(y_test, y_pred_manual)
    manual_accuracy_scores.append(accuracy)
    print(f"K={k}, 手动实现准确率={accuracy:.4f}")  # 打印每个 K 的进度

print("K 值测试完成。")
print("-" * 30)

# --- 5. 确定手动实现的最优 K 值 ---
# 找到最高准确率对应的索引
best_manual_accuracy_index = np.argmax(manual_accuracy_scores)
# 通过索引找到最优的 K 值
best_manual_k = k_range[best_manual_accuracy_index]
best_manual_accuracy = manual_accuracy_scores[best_manual_accuracy_index]

print(f"测试的 K 值范围: {list(k_range)}")
formatted_manual_scores = [f"{acc:.4f}" for acc in manual_accuracy_scores]
print(f"每个 K 值对应的手动实现准确率: {formatted_manual_scores}")
print("-" * 30)
print(f"\手动实现找到的最优 K 值是: {best_manual_k}")
print(f"使用 K={best_manual_k} 时，手动实现的最高准确率为: {best_manual_accuracy:.4f}")
print("-" * 30)

# --- 6. 可视化结果 ---
plt.figure(figsize=(12, 6))
# 使用不同的标记和颜色区分手动实现的结果
plt.plot(
    k_range,
    manual_accuracy_scores,
    color="green",
    linestyle="solid",
    marker="s",
    markerfacecolor="orange",
    markersize=8,
    label="Manual KNN Accuracy",
)
plt.title("鸢尾花数据集 手动实现 k-NN 的 K 值与准确率关系图")
plt.xlabel("K 值 (邻居数量)")
plt.ylabel("测试集准确率 (手动实现)")
plt.xticks(k_range)
plt.grid(True)
plt.legend()  # 显示图例

# 标记最优 K 值点
plt.annotate(
    f"最优 K={best_manual_k}\n准确率={best_manual_accuracy:.4f}",
    xy=(best_manual_k, best_manual_accuracy),
    xytext=(best_manual_k + 1, best_manual_accuracy - 0.02),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
)

plt.show()
