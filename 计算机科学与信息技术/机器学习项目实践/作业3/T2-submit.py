import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

diabetes_df = pd.read_csv("diabetes.csv")

print("\n数据信息:")
diabetes_df.info()

X = diabetes_df.drop("Outcome", axis=1)
y = diabetes_df["Outcome"]
feature_names = X.columns.tolist()
target_names = ["No Diabetes", "Diabetes"]

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

print("\n替换 0 值为 NaN 后，每列的 NaN 数量:")
print(X.isnull().sum())

# 使用中位数填充 NaN (因为这些特征可能偏态分布，中位数比均值更稳健)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 将填充后的数据转回 DataFrame，保留列名
X = pd.DataFrame(X_imputed, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


k = 11
knn = KNeighborsClassifier(n_neighbors=k, weights="distance")  # 尝试加权
knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

print("\n")
print(f"准确率: {accuracy:.4f}")
print("\n混淆矩阵:")

print(conf_matrix)

print("\n分类报告:")
print(class_report)
