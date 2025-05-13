import csv
import random
from datetime import datetime, timedelta
import numpy as np

# 设置随机种子保证可重复性
random.seed(2023)
np.random.seed(2023)


def generate_vision_data(num_records):
    """生成包含校园环境和家庭用眼的综合视力数据"""
    data = []

    for i in range(1, num_records + 1):
        # 学生基础信息
        student_id = f"VIS_{20230000 + i}"
        gender = random.choices(["男", "女"], [0.52, 0.48])[0]
        age = random.randint(6, 18)
        grade = f"小学{age - 6}年级" if age <= 12 else f"初中{age - 12}年级" if age <= 15 else f"高中{age - 15}年级"

        # 视力状况（考虑年龄增长近视概率上升）
        myopia_prob = min(0.3 + (age - 6) * 0.07, 0.85)  # 6岁30%，每年增加7%
        has_myopia = np.random.choice([True, False], p=[myopia_prob, 1 - myopia_prob])
        myopia_degree = round(np.clip(np.random.normal(-3.0, 1.5), -6.0, -0.5), 1) if has_myopia else 0.0

        # 校园视觉环境（考虑现代学校改造现状）
        classroom_data = {
            "教室照度(LUX)": random.choices([300, 250, 180, 150], weights=[0.5, 0.3, 0.15, 0.05])[0],
            "黑板反光情况": random.choices(["无反射", "轻微反光", "明显反光"], [0.6, 0.3, 0.1]),
            "课桌椅匹配度": random.choices(["完全匹配", "偏高5cm", "偏低3cm"], [0.45, 0.3, 0.25]),
            "绿植覆盖率": random.choices([">30%", "20%-30%", "<20%"], [0.4, 0.4, 0.2]),
            "坐姿矫正频率": random.randint(0, 10)  # 每日教师提醒次数
        }

        # 家庭用眼习惯（考虑现代生活方式）
        home_data = {
            "每日屏幕时间": np.random.poisson(3.5),  # 泊松分布模拟使用频率
            "阅读距离监测": random.choices(["智能台灯监测", "家长提醒", "无监控"], [0.2, 0.6, 0.2]),
            "夜间用眼模式": random.choices(["护眼模式", "普通模式", "黑暗环境"], [0.3, 0.5, 0.2]),
            "户外活动时长": max(0, int(np.random.normal(60, 30))),  # 正态分布
            "眼保健操频率": random.choices(["每天2次", "每天1次", "偶尔做"], [0.4, 0.4, 0.2])
        }

        # 复合特征生成
        home_data["连续用眼超标率"] = f"{min(100, int((home_data['每日屏幕时间'] / 4) * 100))}%"  # 假设4小时为合理值
        classroom_data["视觉舒适评分"] = random.randint(60, 98)  # 模拟学生问卷评分

        # 生成时间序列数据
        last_check_date = datetime.now() - timedelta(days=random.randint(30, 720))

        record = {
            # 基础信息
            "学籍编号": student_id,
            "性别": gender,
            "年龄": age,
            "在读年级": grade,

            # 视力健康
            "是否近视": "是" if has_myopia else "否",
            "屈光度数": myopia_degree,
            "矫正视力": round(random.uniform(0.6, 1.2), 1) if has_myopia else 1.0,

            # 校园环境
            **classroom_data,

            # 家庭习惯
            **home_data,

            # 时间维度
            "数据采集日期": last_check_date.strftime("%Y-%m-%d"),
            "下次检查日期": (last_check_date + timedelta(days=180)).strftime("%Y-%m-%d")
        }

        data.append(record)

    return data


# 生成1000条样本数据
vision_data = generate_vision_data(1000)

# 定义CSV文件字段顺序
fieldnames = [
    "学籍编号", "性别", "年龄", "在读年级",
    "是否近视", "屈光度数", "矫正视力",
    "教室照度(LUX)", "黑板反光情况", "课桌椅匹配度", "绿植覆盖率", "坐姿矫正频率", "视觉舒适评分",
    "每日屏幕时间", "阅读距离监测", "夜间用眼模式", "户外活动时长", "眼保健操频率",
    "连续用眼超标率", "数据采集日期", "下次检查日期"
]

# 写入CSV文件
with open("vision_health_dataset.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(vision_data)

print("数据集生成完成，包含1000条视力健康记录")