# coding:UTF-8
__author__ = 'cq'

import traceback
from http import server

import joblib
import numpy as np
import torch
from pyts.preprocessing import scaler
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from app import app
from flask import render_template, request, flash, redirect, url_for, send_from_directory, Blueprint, jsonify, send_file
from app.utils.upload_tools import get_filetype, random_name, img_allowed_file
import os
from PIL import Image
from model import MyopiaPredictor
import torchvision.transforms as transforms

# 全局变量
app.config['UPLOAD_FOLDER'] = 'imgUpload'
app.config['state_dict_path'] = './modelSave/best_model.pth'
device = 'cpu'


class Config:
    seq_length = 3  # 每个患者的时间步长
    dynamic_features = ['ser', 'axial_length', 'k1', 'al_ratio', 'screen_time']
    static_features = ['gender', 'genetic_risk']
    num_classes = 4  # 新增类别数
    test_size = 0.2
    batch_size = 32
    hidden_size = 128  # 增大隐藏层维度
    num_layers = 2
    dropout = 0.5
    lr = 1e-3
    epochs = 50


class preConfig:
    seq_length = 3  # 每个患者的时间步长
    dynamic_features = ['ser', 'axial_length', 'k1', 'al_ratio', 'screen_time']
    static_features = ['gender', 'genetic_risk']


School_DEFAULT_FILE = './dataset/eye_health_data.xlsx'
# Family_DEFAULT_FILE = './dataset/family_eye_habits.xlsx'
Family_DEFAULT_FILE = r'D:\pycharmWorkspace\Eyes\app\dataset\family_eye_habits.xlsx'

# 配置文件上传
ALLOWED_EXTENSIONS = {'xlsx'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 初始化模型并加载权重
def load_pytorch_model(model_path):
    # 必须与训练时的参数一致
    model = MyopiaPredictor(
        input_size=len(Config.dynamic_features),
        static_size=len(Config.static_features),
        hidden_size=Config.hidden_size,
        num_layers=Config.num_layers,
        num_classes=Config.num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


@app.route('/', methods=['POST', 'GET'])
@app.route('/index/', methods=['POST', 'GET'])
def index():
    return render_template('./home/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("[DEBUG] 进入预测函数")
    try:
        # 1. 获取请求数据并验证
        data = request.get_json()
        print("[DEBUG] 原始请求数据:", data)

        if not data:
            print("[ERROR] 空数据请求")
            return jsonify({"error": "No input data"}), 400

        # 2. 类型强制转换
        print("[DEBUG] 执行类型转换...")
        converted_data = {
            # 数值型字段
            'axial_length': float(data.get('axial_length', 0)),
            'ser': float(data.get('ser', 0)),
            'k1': float(data.get('k1', 0)),
            'al_ratio': float(data.get('al_ratio', 0)),
            'screen_time': float(data.get('screen_time', 0)),

            # 分类字段编码
            'gender': 0 if data.get('gender') == 'male' else 1,

            # 整型字段
            'genetic_risk': int(data.get('genetic_risk', 0)),
            'observation_period': int(data.get('observation_period', 12))
        }

        # 3. 数据预处理强化
        print("[DEBUG] 开始数据预处理...")
        processor = PredictionPreprocessor(Config)

        # 添加维度检查
        dynamic_seq, static_features = processor.process_single(converted_data)
        print(f"[DEBUG] 预处理后类型 - 动态序列: {dynamic_seq.dtype}, 静态特征: {static_features.dtype}")

        # 4. 显式类型转换
        dynamic_seq = dynamic_seq.astype(np.float32)
        static_features = static_features.astype(np.float32)

        # 5. 张量转换保护
        try:
            dynamic_tensor = torch.FloatTensor(dynamic_seq)
            static_tensor = torch.FloatTensor(static_features)
        except TypeError as e:
            print(f"[ERROR] 张量转换失败: {str(e)}")
            print(f"动态序列样本: {dynamic_seq[:2]}")
            print(f"静态特征样本: {static_features[:2]}")
            return jsonify({"error": "数据格式转换失败"}), 400

        # 6. 合并输入验证
        if dynamic_tensor.dim() != 3 or static_tensor.dim() != 2:
            print(f"[ERROR] 输入维度异常 - 动态: {dynamic_tensor.shape} 静态: {static_tensor.shape}")
            raise ValueError("输入维度不符合要求")

            # 前向传播
        model = load_pytorch_model(app.config['state_dict_path'])
        with torch.no_grad():
            print(f"[DEBUG] 输入维度 - 动态: {dynamic_tensor.shape}, 静态: {static_tensor.shape}")

            # 模型前向传播
            logits = model(dynamic_tensor, static_tensor)

            # 获取预测概率
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(logits, dim=1))

        print("[DEBUG] 预测完成 - 类别: {} 概率分布: {}".format(predicted_class, probabilities))

        return jsonify({
            "code": predicted_class
        })



    except Exception as e:
        print(f"[CRITICAL] 异常堆栈: {traceback.format_exc()}")
    return jsonify({"error": f"处理失败: {str(e)}"}), 500


class PredictionPreprocessor:
    def __init__(self, config):
        """
        :param config: 包含以下属性的配置对象
            - seq_length: 时间序列长度（训练时设定的滑动窗口大小）
            - dynamic_features: 动态特征列表（顺序必须与训练时一致）
            - static_features: 静态特征列表（顺序必须与训练时一致）
        """
        self.scaler = joblib.load("./modelSave/standard_scaler.pkl")  # 加载训练时的scaler
        self.config = config

    def process_single(self, request_data):
        """处理单条预测请求"""
        # 构造动态特征序列
        dynamic_seq = self._build_dynamic_sequence(request_data)

        # 处理静态特征
        static_features = np.array([
            request_data[feat] for feat in self.config.static_features
        ]).reshape(1, -1)

        return dynamic_seq, static_features

    def _build_dynamic_sequence(self, data):
        """构建符合模型输入要求的动态特征序列"""
        # 从请求中提取动态特征（保持顺序一致）
        current_step = np.array([
            data[feat] for feat in self.config.dynamic_features
        ]).reshape(1, -1)

        # 自动构建滑动窗口（示例逻辑，根据实际需求修改）
        if not hasattr(self, 'history_buffer'):
            self.history_buffer = np.repeat(current_step, self.config.seq_length - 1, axis=0)

        # 合并历史数据
        full_sequence = np.concatenate([self.history_buffer, current_step])

        # 更新缓冲区（FIFO）
        self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=0)
        self.history_buffer[-1] = current_step[0]

        # 标准化处理
        scaled = self.scaler.transform(full_sequence)
        return scaled.reshape(1, self.config.seq_length, -1)  # 添加batch维度


def build_response(risk, input_data):
    """构建前端需要的响应结构"""
    return {
        "risk_percentage": round(risk, 1),
        "risk_level": get_risk_level(risk),
        "trend_data": generate_trend(risk),
        "recommendations": generate_recommendations(input_data, risk)
    }


def get_risk_level(percent):
    if percent < 30:
        return {"level": "低风险", "color": "#00B894"}
    elif 30 <= percent < 60:
        return {"level": "中风险", "color": "#FDCB6E"}
    else:
        return {"level": "高风险", "color": "#D63031"}


def generate_trend(base_risk):
    return {
        "timeline": ["当前", "3月", "6月", "9月", "12月"],
        "values": [round(base_risk * (1 + i * 0.1), 1) for i in range(5)]
    }


def generate_recommendations(data, risk):
    """生成个性化建议"""
    recs = []

    if float(data.get('screen_time', 0)) > 4:
        recs.append({
            "icon": "fa-clock",
            "title": "屏幕时间",
            "content": "建议每日屏幕使用时间减少至2小时以内"
        })

    if risk > 60:
        recs.append({
            "icon": "fa-eye",
            "title": "专业检查",
            "content": "建议每2个月进行专业眼科检查"
        })

    # 添加通用建议
    recs.extend([
        {
            "icon": "fa-sun",
            "title": "户外活动",
            "content": "每日保证至少2小时户外光照"
        },
        {
            "icon": "fa-book",
            "title": "用眼习惯",
            "content": "遵循20-20-20护眼法则"
        }
    ])

    return recs


###################################### 校园环境优化，特征分析####################################################
@app.route('/analyze/echarts', methods=['get', 'post'])
def analyze_for_echarts():
    """生成特征重要性数据"""
    try:
        # 处理文件上传
        if request.method == 'POST':
            # 检查是否有文件部分
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['file']
            # 如果用户没有选择文件
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if file and allowed_file(file.filename):
                # 使用上传的文件
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Invalid file type"}), 400
        else:
            # 读取并验证数据
            df = pd.read_excel(School_DEFAULT_FILE)

        required_columns = ['教室灯光类型', '桌椅高度可调', '桌椅布局',
                            '每日阅读时间(小时)', '电子产品使用时间(小时)',
                            '正确阅读姿势', '每日户外活动(小时)', '视力正常']
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            return jsonify({"error": f"缺少必要字段: {missing}"}), 400

        # 预处理
        le = LabelEncoder()
        df['教室灯光类型'] = le.fit_transform(df['教室灯光类型'])
        df['桌椅布局'] = le.fit_transform(df['桌椅布局'])

        # 准备数据
        X = df[required_columns[:-1]]  # 排除目标列
        y = df['视力正常']

        # 训练模型
        model = RandomForestClassifier(n_estimators=150)
        model.fit(X, y)

        # 处理特征重要性
        features = X.columns.tolist()
        importances = model.feature_importances_.round(4).tolist()

        # 按重要性排序
        sorted_data = sorted(zip(features, importances),
                             key=lambda x: x[1],
                             reverse=True)

        # 构建返回数据格式
        result = [{"feature": feature, "importance": float(importance)}
                  for feature, importance in sorted_data]

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



@app.route('/analysis/relation_map', methods=['GET', 'POST'])
def relation_analysis():
    """支持文件上传的多维度关系分析接口"""
    try:
        # 处理文件上传
        if request.method == 'POST':
            # 检查是否有文件部分
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['file']
            # 如果用户没有选择文件
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if file and allowed_file(file.filename):
                # 使用上传的文件
                data = pd.read_excel(file)
            else:
                return jsonify({"error": "Invalid file type"}), 400
        else:
            # GET请求使用默认文件
            data = pd.read_excel(School_DEFAULT_FILE)

        # 验证数据列
        required_columns = ['教室灯光类型', '视力正常', '电子产品使用时间(小时)',
                            '每日户外活动(小时)', '每日阅读时间(小时)']
        if not all(col in data.columns for col in required_columns):
            return jsonify({
                "error": f"文件缺少必要字段，需要包含：{', '.join(required_columns)}"
            }), 400

        # 分析逻辑
        # 灯光类型与视力关系
        light_vision = data.groupby('教室灯光类型')['视力正常'].mean().reset_index()

        # 电子产品使用分布
        screen_dist = {
            "0-2小时": len(data[data['电子产品使用时间(小时)'] <= 2]),
            "2-4小时": len(data[(data['电子产品使用时间(小时)'] > 2) &
                                (data['电子产品使用时间(小时)'] <= 4)]),
            "4小时以上": len(data[data['电子产品使用时间(小时)'] > 4])
        }

        # 户外活动与阅读时间相关性
        corr_value = np.corrcoef(data['每日户外活动(小时)'],
                                 data['每日阅读时间(小时)'])[0, 1]

        return jsonify({
            "lightVision": {
                "categories": light_vision['教室灯光类型'].tolist(),
                "values": light_vision['视力正常'].round(2).tolist()
            },
            "screenTime": {
                "legend": list(screen_dist.keys()),
                "data": list(screen_dist.values())
            },
            "activityCorrelation": round(corr_value, 2)
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "上传的文件为空或格式不正确"}), 400
    except Exception as e:
        app.logger.error(f"分析错误: {str(e)}")
        return jsonify({"error": "服务器内部错误"}), 500


@app.route('/download/school_file', methods=['GET'])
def download_school_file():
    try:
        # 检查文件是否存在
        if not os.path.exists(School_DEFAULT_FILE):
            return {"error": "School eye habits file not found"}, 404

        # 使用send_file发送文件
        return send_file(
            School_DEFAULT_FILE,
            as_attachment=True,
            download_name='School_Eye_Habits.xlsx',  # 下载时的默认文件名
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        return {"error": str(e)}, 500


#######family_eye########################################################################################
def analyze_dataset(df):
    """核心数据分析逻辑"""
    # 基础统计
    basic_stats = {
        'total_students': len(df),
        'vision_normal_rate': round(df['视力正常'].mean(), 4),
        'avg_reading': round(df['每日阅读时间(小时)'].mean(), 2),
        'avg_screen_time': round(df['电子产品使用时间(小时)'].mean(), 2),
        'avg_outdoor': round(df['每日户外活动(小时)'].mean(), 2)
    }

    # 灯光类型分析
    light_analysis = df.groupby('教室灯光类型').agg(
        vision_rate=('视力正常', 'mean'),
        avg_reading=('每日阅读时间(小时)', 'mean')
    ).round(4).to_dict()

    # 屏幕时间分布
    screen_bins = [0, 2, 4, 6, np.inf]
    screen_labels = ['0-2h', '2-4h', '4-6h', '6+h']
    screen_dist = df['电子产品使用时间(小时)'].value_counts(
        bins=screen_bins,
        sort=False
    ).to_dict()

    # 桌椅布局分析
    layout_analysis = df.groupby('桌椅布局').agg(
        vision_rate=('视力正常', 'mean'),
        posture_rate=('正确阅读姿势', 'mean')
    ).round(4).to_dict()

    # 特征重要性分析
    X = pd.get_dummies(df.drop(['视力正常', '学生姓名'], axis=1, errors='ignore'))
    y = df['视力正常']

    model = RandomForestClassifier()
    model.fit(X, y)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).round(4).to_dict()

    return {
        'basic_stats': basic_stats,
        'light_analysis': light_analysis,
        'screen_distribution': {
            'bins': screen_labels,
            'counts': [int(v) for v in screen_dist.values()]
        },
        'layout_analysis': layout_analysis,
        'feature_importance': feature_importance
    }


@app.route('/api/analysis', methods=['Get', 'POST'])
def handle_analysis():
    print('访问分析接口')
    try:
        if request.files.get('file'):
            df = pd.read_excel(request.files['file'])
        else:
            df = pd.read_excel(School_DEFAULT_FILE)

        # 验证必要字段
        required_cols = ['教室灯光类型', '视力正常', '电子产品使用时间(小时)',
                         '每日户外活动(小时)', '正确阅读姿势', '桌椅布局']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({
                "error": f"缺少必要字段: {', '.join(missing)}"
            }), 400

        # 执行分析
        analysis_result = analyze_dataset(df)

        # 构建前端友好格式
        formatted_data = {
            'light_data': {
                'categories': list(analysis_result['light_analysis']['vision_rate'].keys()),
                'vision_rates': list(analysis_result['light_analysis']['vision_rate'].values()),
                'reading_times': list(analysis_result['light_analysis']['avg_reading'].values())
            },
            'screen_data': analysis_result['screen_distribution'],
            'layout_data': analysis_result['layout_analysis'],
            'feature_importance': analysis_result['feature_importance']
        }

        return jsonify(formatted_data)

    except Exception as e:
        return jsonify({
            "error": f"分析失败: {str(e)}"
        }), 500


@app.route('/download/family_file', methods=['GET'])
def download_family_file():
    try:
        # 检查文件是否存在
        if not os.path.exists(Family_DEFAULT_FILE):
            return {"error": "Family eye habits file not found"}, 404

        # 使用send_file发送文件
        return send_file(
            Family_DEFAULT_FILE,
            as_attachment=True,
            download_name='Family_Eye_Habits.xlsx',  # 下载时的默认文件名
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dataType')
def dataType():
    return render_template('./dataFirst/index.html')


@app.route('/dataType_2_1')
def dataType_2_1():
    return render_template('./dataSecond/index.html')


@app.route('/dataType_3_1')
def dataType_3_1():
    return render_template('./dataThird/index.html')


@app.route('/familyEye')
def familyEye():
    return render_template('./Famliy_data/index.html')


@app.route('/schoolEye')
def schoolEye():
    return render_template('./School_data/index.html')


# ----------------------------------------------错误处理页面---------------------------------------------
@app.errorhandler(404)
def internal_error(error):
    return render_template('./error/404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('./error/500.html'), 500
