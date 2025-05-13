import pandas as pd
import numpy as np
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化Faker生成中文数据
fake = Faker('zh_CN')

# 配置参数
np.random.seed(2023)
SAMPLE_SIZE = 1000


# 生成模拟数据
def generate_eye_data(size):
    data = []
    for _ in range(size):
        # 教室环境特征
        classroom_light = np.random.choice(['LED', '荧光灯', '混合'], p=[0.6, 0.3, 0.1])
        desk_adjustable = np.random.choice([1, 0], p=[0.7, 0.3])
        desk_layout = np.random.choice(['传统排列', '小组讨论式', '灵活组合'], p=[0.5, 0.3, 0.2])

        # 家庭用眼习惯
        reading_time = np.clip(np.random.normal(3.5, 1), 1, 6)  # 每天阅读时间（小时）
        screen_time = np.clip(np.random.exponential(2), 0, 8)  # 电子产品使用时间
        correct_posture = 1 if np.random.rand() > 0.3 else 0  # 正确姿势概率
        outdoor_time = np.clip(np.random.normal(1.5, 1), 0, 4)  # 户外活动时间

        # 构建目标变量逻辑关系
        vision_risk = 0
        vision_risk += screen_time * 0.3
        vision_risk -= outdoor_time * 0.4
        vision_risk += (reading_time > 4) * 0.5
        vision_risk += (classroom_light == '荧光灯') * 0.2
        vision_risk += (desk_adjustable == 0) * 0.3
        vision_prob = 1 / (1 + np.exp(-vision_risk))
        vision_normal = 1 if np.random.rand() > vision_prob else 0

        data.append([
            fake.name(),  # 学生姓名
            classroom_light,
            desk_adjustable,
            desk_layout,
            round(reading_time, 1),
            round(screen_time, 1),
            correct_posture,
            round(outdoor_time, 1),
            vision_normal
        ])

    return pd.DataFrame(data, columns=[
        '学生姓名',
        '教室灯光类型',
        '桌椅高度可调',
        '桌椅布局',
        '每日阅读时间(小时)',
        '电子产品使用时间(小时)',
        '正确阅读姿势',
        '每日户外活动(小时)',
        '视力正常'
    ])


# 生成并保存数据
df = generate_eye_data(SAMPLE_SIZE)
df.to_excel('eye_health_data.xlsx', index=False)
print("数据已保存至 eye_health_data.xlsx")

# 读取数据预处理
df = pd.read_excel('eye_health_data.xlsx')

# 特征编码处理
le = LabelEncoder()
categorical_cols = ['教室灯光类型', '桌椅布局']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 划分特征和目标
X = df.drop(['学生姓名', '视力正常'], axis=1)
y = df['视力正常']

# 特征重要性分析
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 可视化设置
plt.figure(figsize=(12, 8))
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性图表
plt.title('特征重要性排序')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('相对重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("特征重要性可视化已保存至 feature_importance.png")

# 保存带特征名称的重要性数据
feature_importance = pd.DataFrame({'特征': features, '重要性': importances})
feature_importance.sort_values('重要性', ascending=False).to_excel('feature_importance.xlsx', index=False)

# 显示数据样例
print("\n生成数据样例：")
print(df.head(3))

# 显示分析结果
print("\n特征重要性排序：")
print(feature_importance.sort_values('重要性', ascending=False))