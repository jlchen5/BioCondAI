# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 数据预处理
# 假设你有一个名为 'condensate_data.csv' 的CSV文件，包含了生物小分子的特征和它们是否形成凝聚体的标签
data = pd.read_csv('condensate_data.csv')
X = data.drop('condensate_label', axis=1)  # 特征
y = data['condensate_label']  # 标签

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
# 使用随机森林作为示例模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# 保存模型，供后续使用
import joblib
joblib.dump(model, 'biocondai_model.pkl')
