import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import math

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def knn_predict_single(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((y_train[i], dist))

    distances.sort(key=lambda x: x[1])

    neighbors_labels = []
    k_actual = min(k, len(distances))
    if k_actual <= 0:
        return -1

    for i in range(k_actual):
        neighbors_labels.append(distances[i][0])

    most_common = Counter(neighbors_labels).most_common(1)

    if not most_common:
        return -1

    return most_common[0][0]


k_range = range(1, 26)
manual_accuracy_scores = []

for k in k_range:
    y_pred_manual = []
    for test_point in X_test:
        prediction = knn_predict_single(X_train, y_train, test_point, k)
        y_pred_manual.append(prediction)

    accuracy = accuracy_score(y_test, y_pred_manual)
    manual_accuracy_scores.append(accuracy)
    print(f"K={k}, 准确率={accuracy:.4f}")


best_manual_accuracy_index = np.argmax(manual_accuracy_scores)
best_manual_k = k_range[best_manual_accuracy_index]
best_manual_accuracy = manual_accuracy_scores[best_manual_accuracy_index]

formatted_manual_scores = [f"{acc:.4f}" for acc in manual_accuracy_scores]

plt.figure(figsize=(12, 6))
plt.plot(
    k_range,
    manual_accuracy_scores,
    color="green",
    linestyle="solid",
    marker="s",
    markerfacecolor="orange",
    markersize=8,
    label="KNN Accuracy",
)
plt.title("鸢尾花数据集 k-NN K 值与准确率关系图")
plt.xlabel("K 值 ")
plt.ylabel("测试集准确率")
plt.xticks(k_range)
plt.grid(True)
plt.legend()

plt.annotate(
    f"最优 K={best_manual_k}\n准确率={best_manual_accuracy:.4f}",
    xy=(best_manual_k, best_manual_accuracy),
    xytext=(best_manual_k + 1, best_manual_accuracy - 0.02),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
)

plt.show()
