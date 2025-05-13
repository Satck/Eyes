import pandas as pd
import numpy as np
from scipy.stats import truncnorm


def generate_myopia_multi_class_csv(n_patients=10000, n_timesteps=3, output_path="balanced_class.csv"):
    np.random.seed(42)

    # 初始化容器
    data = []
    class_weights = {
        0: 0.25,  # 非近视
        1: 0.25,  # 轻度
        2: 0.25,  # 中度
        3: 0.25  # 重度
    }

    # 参数范围映射表
    param_ranges = {
        0: {'ser_range': (-0.5, 1.0), 'genetic_range': (0.3, 0.5), 'axial_base': (22.5, 23.5)},
        1: {'ser_range': (-3.0, -0.5), 'genetic_range': (0.4, 0.6), 'axial_base': (23.5, 24.0)},
        2: {'ser_range': (-6.0, -3.0), 'genetic_range': (0.5, 0.7), 'axial_base': (24.0, 24.5)},
        3: {'ser_range': (-9.0, -6.0), 'genetic_range': (0.6, 0.9), 'axial_base': (24.5, 25.5)}
    }

    # 为每个类别生成样本
    for cls, weight in class_weights.items():
        n_cls = int(n_patients * weight)

        for pid in range(n_cls):
            # 获取类别特定参数范围
            params = param_ranges[cls]

            # 基础参数生成
            genetic = np.random.uniform(*params['genetic_range'])
            axial_base = np.random.uniform(*params['axial_base'])
            screen_time_base = np.clip(3 + genetic * 2, 2, 6)
            base_age = np.random.randint(6, 13)

            # 动态参数控制
            progression_rate = 0.1 + (genetic * 0.2) + (screen_time_base * 0.05)
            ser_decay = (params['ser_range'][0] - params['ser_range'][1]) / n_timesteps

            # 时间序列生成
            for t in range(n_timesteps):
                # 计算当前参数值
                current_ser = params['ser_range'][1] + (ser_decay * (t + 1))
                axial_length = axial_base + (0.1 * progression_rate * (t + 1))

                # 添加随机波动
                current_ser += np.random.normal(0, 0.2)
                axial_length += np.random.normal(0, 0.05)

                # 构建记录
                record = {
                    "patient_id": f"C{cls}_P{pid:04d}",
                    "time_step": t,
                    "age": base_age + t,
                    "ser": current_ser,
                    "axial_length": axial_length,
                    "k1": 43.5 - 0.1 * current_ser + np.random.normal(0, 0.1),
                    "al_ratio": axial_length / (43.5 * 0.1),
                    "screen_time": screen_time_base + t * 0.3,
                    "gender": np.random.choice([0, 1]),
                    "genetic_risk": genetic,
                    "label": cls
                }
                data.append(record)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 验证关键参数范围
    assert df["axial_length"].between(22.0, 26.0).all(), "眼轴长度异常"
    assert df["ser"].between(-10.0, 3.0).all(), "等效球镜异常"

    # 保存数据
    df.to_csv(output_path, index=False)

    # 验证类别分布
    final_labels = df.groupby('patient_id')['label'].last()
    print("最终类别分布：")
    print(final_labels.value_counts(normalize=True))

    return df


# 生成平衡数据集（10000个样本）
generate_myopia_multi_class_csv(n_patients=10000)