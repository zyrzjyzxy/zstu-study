import numpy as np
import pandas as pd

# 从 ucimlrepo 导入数据集（如果安装了该库且网络可用）
# 或者直接从本地 CSV 加载
# from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer  # 用于处理缺失值（0值）

# --- 数据加载与准备 ---
print("--- 1. 加载和准备数据 ---")

# 尝试从本地 CSV 加载
try:
    # **重要:** 确保 'diabetes.csv' 文件在你的工作目录中，或者提供完整路径
    diabetes_df = pd.read_csv("diabetes.csv")
    print("成功从本地 'diabetes.csv' 加载数据。")

    # 查看数据基本信息
    print("\n数据前5行:")
    print(diabetes_df.head())
    print("\n数据信息:")
    diabetes_df.info()
    print("\n数据描述统计:")
    print(diabetes_df.describe())

    # 分离特征 (X) 和目标 (y)
    # 假设 'Outcome' 是目标列，其余为特征列
    X = diabetes_df.drop("Outcome", axis=1)
    y = diabetes_df["Outcome"]
    feature_names = X.columns.tolist()
    target_names = ["No Diabetes", "Diabetes"]  # 0: No Diabetes, 1: Diabetes

except FileNotFoundError:
    print("错误: 'diabetes.csv' 文件未找到。")
    print(
        "请确保文件存在于当前目录，或使用下面的代码尝试从网络获取（需要安装 ucimlrepo）。"
    )
    # 备选：尝试从 ucimlrepo 获取 (需要 pip install ucimlrepo)
    # try:
    #     print("\n尝试从 ucimlrepo 获取 Pima Indians Diabetes 数据集...")
    #     pima_indians_diabetes = fetch_ucirepo(id=22)
    #     X = pima_indians_diabetes.data.features
    #     y = pima_indians_diabetes.data.targets['Outcome'] # 确保目标列名称正确
    #     feature_names = X.columns.tolist()
    #     target_names = ['No Diabetes', 'Diabetes']
    #     print("成功从 ucimlrepo 获取数据。")
    # except Exception as e:
    #     print(f"从 ucimlrepo 获取数据失败: {e}")
    #     print("无法加载数据，请检查网络连接或手动下载 CSV 文件。")
    exit()  # 如果无法加载数据，则退出

print(f"\n数据集样本数: {X.shape[0]}")
print(f"数据集特征数: {X.shape[1]}")
print(f"特征名称: {feature_names}")
print(f"目标类别: {target_names} (0={target_names[0]}, 1={target_names[1]})")
print("-" * 30 + "\n")


# --- (可选但强烈推荐) 处理代表缺失的 0 值 ---
print("--- 1b. (可选) 处理代表缺失的 0 值 ---")
# 这些列中的 0 值不合理，应视为缺失
cols_with_zeros_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
# 将这些列中的 0 替换为 NaN
X[cols_with_zeros_as_missing] = X[cols_with_zeros_as_missing].replace(0, np.nan)

print("替换 0 值为 NaN 后，每列的 NaN 数量:")
print(X.isnull().sum())

# 使用中位数填充 NaN (因为这些特征可能偏态分布，中位数比均值更稳健)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 将填充后的数据转回 DataFrame，保留列名
X = pd.DataFrame(X_imputed, columns=feature_names)
print("\n使用中位数填充 NaN 完成。")
print("处理后的数据前5行:")
print(X.head())
print("-" * 30 + "\n")
# --- 处理结束 ---


# --- 2. 划分训练集和测试集 ---
print("--- 2. 划分训练集和测试集 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print("-" * 30 + "\n")

# --- 3. 特征标准化 ---
print("--- 3. 特征标准化 ---")
scaler = StandardScaler()
# 注意：fit_transform 应用在 DataFrame 上会丢失列名，如需保留需重新构造
# 或者直接在 NumPy 数组上操作
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("标准化完成。")
print("-" * 30 + "\n")

# --- 4. 初始化并训练 kNN 模型 ---
print("--- 4. 训练 kNN 模型 ---")
# 可以尝试不同的 K 值，或者使用 GridSearchCV 寻找最佳 K
k = 11  # Pima 数据集通常 K 较大时效果可能更好，需要调优
knn = KNeighborsClassifier(n_neighbors=k, weights="distance")  # 尝试加权
knn.fit(X_train_scaled, y_train)
print(f"kNN 模型训练完成 (k={k}, weights='distance')。")
print("-" * 30 + "\n")

# --- 5. 在测试集上进行预测 ---
print("--- 5. 进行预测 ---")
y_pred = knn.predict(X_test_scaled)
print("预测完成。")
print("-" * 30 + "\n")

# --- 6. 评估模型性能 ---
print("--- 6. 评估模型 ---")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

print(f"模型准确率 (Accuracy): {accuracy:.4f}")
print("\n混淆矩阵 (Confusion Matrix):")
# [[TN, FP],
#  [FN, TP]]
# TN: 真实为 0 (No Diabetes), 预测为 0
# FP: 真实为 0 (No Diabetes), 预测为 1
# FN: 真实为 1 (Diabetes),    预测为 0
# TP: 真实为 1 (Diabetes),    预测为 1
print(conf_matrix)

print("\n分类报告 (Classification Report):")
print(class_report)
# 特别关注 'Diabetes' (标签1) 的 Recall，因为它通常是更关心的指标（漏诊）
print("-" * 30 + "\n")

# --- 7. 提升建议回顾 ---
print("--- 7. 如何进一步提升效果? ---")
print(
    "1.  **精细调优 K 值:** 使用 GridSearchCV 或 RandomizedSearchCV 结合交叉验证寻找最优 K。"
)
print("2.  **尝试不同距离度量:** 如曼哈顿距离 (`metric='manhattan'`)。")
print("3.  **特征工程:** 是否可以创建新的有意义的特征？（例如，BMI 分级）。")
print("4.  **特征选择:** 移除冗余或不重要的特征。")
print("5.  **更复杂的缺失值处理:** 尝试其他插补方法。")
print(
    "6.  **处理类别不平衡:** 如果需要，使用 SMOTE 等过采样技术（尤其是在模型对少数类预测效果差时）。"
)
print("-" * 30 + "\n")
