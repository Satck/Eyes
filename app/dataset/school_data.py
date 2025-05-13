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