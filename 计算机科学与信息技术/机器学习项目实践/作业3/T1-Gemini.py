# 导入所需的库
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 添加中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # MacOS 的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# --- 1. 加载鸢尾花数据集 ---
iris = load_iris()
X = iris.data  # 特征数据 (花萼长度/宽度, 花瓣长度/宽度)
y = iris.target  # 目标标签 (鸢尾花的种类: 0, 1, 2)

print(f"数据集包含 {X.shape[0]} 个样本和 {X.shape[1]} 个特征。")
print(f"目标类别: {iris.target_names}")
print("-" * 30)

# --- 2. 划分训练集和测试集 ---
# test_size=0.3 表示将 30% 的数据用作测试集，70% 用作训练集。
# random_state=42 确保每次运行代码时划分方式都相同，便于结果复现。
# stratify=y 确保在划分时，训练集和测试集中各个类别的比例与原始数据集中大致相同，这对于分类问题很重要。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")
print("-" * 30)

# --- 3. 寻找最优 K 值 ---
# 定义要测试的 K 值范围。通常从 1 开始，到一个合理的上限（例如样本数的平方根或某个固定值）。
# 对于这个小型数据集，我们可以测试 1 到 25。
k_range = range(1, 26)
accuracy_scores = []  # 用于存储每个 K 值对应的测试集准确率

print("开始测试不同的 K 值...")
# 循环遍历 K 值范围
for k in k_range:
    # 创建 KNeighborsClassifier 实例，设置邻居数量为当前的 k
    knn = KNeighborsClassifier(n_neighbors=k)

    # 使用训练数据训练 kNN 模型
    knn.fit(X_train, y_train)

    # 使用训练好的模型对测试数据进行预测
    y_pred = knn.predict(X_test)

    # 计算并记录当前 K 值在测试集上的准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    # print(f"K={k}, 测试集准确率={accuracy:.4f}") # 可以取消注释以查看每个k的得分

print("K 值测试完成。")
print("-" * 30)

# --- 4. 确定最优 K 值 ---
# 找到准确率列表中的最大值对应的索引
best_accuracy_index = np.argmax(accuracy_scores)
# 通过索引找到最优的 K 值 (注意 k_range 从 1 开始，索引从 0 开始)
best_k = k_range[best_accuracy_index]
best_accuracy = accuracy_scores[best_accuracy_index]

print(f"测试的 K 值范围: {list(k_range)}")
# 为了更清晰地展示，格式化准确率输出
formatted_scores = [f"{acc:.4f}" for acc in accuracy_scores]
print(f"每个 K 值对应的准确率: {formatted_scores}")
print("-" * 30)
print(f"\找到的最优 K 值是: {best_k}")
print(f"使用 K={best_k} 时，在测试集上达到的最高准确率为: {best_accuracy:.4f}")
print("-" * 30)

# --- 5. 可视化 K 值与准确率的关系 ---
plt.figure(figsize=(12, 6))
plt.plot(
    k_range,
    accuracy_scores,
    color="red",
    linestyle="dashed",
    marker="o",
    markerfacecolor="blue",
    markersize=10,
)
plt.title("鸢尾花数据集 k-NN 算法的 K 值与准确率关系图")
plt.xlabel("K 值 (邻居数量)")
plt.ylabel("测试集准确率")
plt.xticks(k_range)  # 确保 x 轴显示所有测试过的 K 值
plt.grid(True)  # 添加网格线

# 在图上标记出最优 K 值点
plt.annotate(
    f"最优 K={best_k}\n准确率={best_accuracy:.4f}",
    xy=(best_k, best_accuracy),  # 箭头指向的点
    xytext=(best_k + 1, best_accuracy - 0.02),  # 文本位置 (稍微偏移)
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
)  # 箭头样式

# 显示图表
plt.show()
