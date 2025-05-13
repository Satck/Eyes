import pandas as pd
import numpy as np
from faker import Faker


def generate_eyehabits_data(sample_size=1000, seed=42):
    """生成模拟青少年用眼习惯数据集"""
    np.random.seed(seed)
    fake = Faker('zh_CN')

    # 生成基础数据框架
    data = pd.DataFrame({
        '学生姓名': [fake.name() for _ in range(sample_size)],
        '年龄': np.random.randint(12, 19, sample_size),
        '教室灯光类型': np.random.choice(['LED', '荧光灯', '混合'], sample_size, p=[0.6, 0.3, 0.1]),
        '桌椅高度可调': np.random.choice([1, 0], sample_size, p=[0.7, 0.3]),
        '桌椅布局': np.random.choice(['传统排列', '小组讨论式', '灵活组合'], sample_size, p=[0.5, 0.3, 0.2]),
        '每日阅读时间(小时)': np.round(np.clip(np.random.normal(3.0, 1.2, sample_size), 0.5, 8.0), 2),
        '电子产品使用时间(小时)': np.round(np.clip(np.random.gamma(3, 0.8, sample_size), 0.0, 12.0), 2),
        '正确阅读姿势': np.random.choice([1, 0], sample_size, p=[0.65, 0.35]),
        '每日户外活动(小时)': np.round(np.clip(np.random.exponential(1.2, sample_size), 0.0, 6.0), 2)
    })

    # 生成视力正常标签（带逻辑规则）
    def calculate_vision(row):
        base_prob = 0.7
        if row['正确阅读姿势'] == 1:
            base_prob += 0.15
        if row['电子产品使用时间(小时)'] < 4:
            base_prob += 0.1
        if row['教室灯光类型'] == 'LED':
            base_prob += 0.05
        return 1 if np.random.random() < base_prob else 0

    data['视力正常'] = data.apply(calculate_vision, axis=1)

    # 添加关联特征
    data['总用眼时间(小时)'] = np.round(data['每日阅读时间(小时)'] + data['电子产品使用时间(小时)'], 2)
    data['有效休息比例'] = np.round(np.random.beta(2, 3, sample_size) * data['每日户外活动(小时)'], 2)

    # 调整列顺序
    cols = [
        '学生姓名', '年龄', '教室灯光类型', '桌椅高度可调', '桌椅布局',
        '每日阅读时间(小时)', '电子产品使用时间(小时)', '总用眼时间(小时)',
        '正确阅读姿势', '每日户外活动(小时)', '有效休息比例', '视力正常'
    ]
    return data[cols]



if __name__ == "__main__":
    # 生成测试数据
    df = generate_eyehabits_data()
    # 保存数据
    df.to_excel('family_eye_habits.xlsx', index=False)
