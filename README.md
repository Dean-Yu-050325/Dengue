# Epidemiology-Informed Spatio-Temporal Graph Neural Network for Dengue Prediction

基于流行病学知识的时空图神经网络模型，用于预测台湾地区登革热病例数。

## 模型架构

### 核心组件

1. **空间建模 (GCN)**: 使用图卷积网络捕获城市间的空间相关性
2. **时间建模 (LSTM)**: 使用LSTM捕获时间序列的时序模式
3. **SIS正则化**: 使用SIS (Susceptible-Infected-Susceptible) 动力学模型作为软约束

### 模型特点

- **混合建模**: 数据驱动 + 机制先验
- **软约束**: 不强制遵循流行病学方程，但通过正则化引导
- **可解释性**: 显式的疾病动力学
- **可扩展性**: 支持可学习的SIS参数、城市特定参数、移动性信息图等

## 文件结构

```
.
├── Dengue_Daily_EN.csv          # 原始数据
├── data_preprocessing.py         # 数据预处理脚本
├── model.py                      # 模型定义
├── train.py                      # 训练脚本
└── README.md                     # 说明文档
```

## 快速开始

### 方式1: 一键运行完整管道

```bash
python run_pipeline.py
```

这将自动执行：
1. 数据预处理
2. 模型训练
3. 结果可视化

### 方式2: 分步执行

#### 步骤1: 数据预处理

```bash
python data_preprocessing.py
```

这将：
- 加载原始CSV数据
- 按城市和日期聚合病例数
- 创建时间序列样本
- 划分训练/验证/测试集
- 保存预处理后的数据到 `processed_data.pkl`

#### 步骤2: 训练模型

```bash
python train.py
```

训练过程将：
- 加载预处理后的数据
- 创建全连接图结构
- 初始化模型
- 训练模型（包含SIS正则化）
- 保存最佳模型和训练曲线

#### 步骤3: 可视化结果

```bash
python visualize_results.py
```

这将生成：
- 预测值 vs 真实值对比图
- 时间序列预测图
- 误差分布图

#### 步骤4: 使用模型进行预测

```bash
# 使用测试集数据预测未来7天
python inference.py --checkpoint checkpoints/best_model.pth --days 7 --use_test_data
```

### 3. 模型配置

可以在 `train.py` 中修改以下配置：

```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'lambda_sis': 0.1,          # SIS正则化权重
    'gcn_hidden_dim': 64,
    'gcn_num_layers': 2,
    'lstm_hidden_dim': 128,
    'lstm_num_layers': 2,
    'dropout': 0.1,
    'use_sis': True,
    'learnable_sis_params': True
}
```

## 模型架构详解

### 输入输出

- **输入**: `X ∈ R^(B×w×N)`，其中B是批次大小，w是时间窗口大小，N是城市数量
- **输出**: `Î_{t+1} ∈ R^N`，预测的下一个时间步的病例数

### 损失函数

总损失 = 数据保真度损失 + λ × SIS一致性损失

```
L = L_data + λ × L_SIS
```

其中：
- `L_data = ||Î_{t+1} - I_{t+1}||²`
- `L_SIS = ||Î_{t+1} - I_{t+1}^{SIS}||²`

### SIS动力学模型

```
I_{t+1}^{SIS} = I_t + β(N - I_t)I_t/N - γI_t
```

其中：
- `I_t`: 当前感染比例
- `β`: 感染率
- `γ`: 恢复率
- `N`: 总人口（归一化为1）

## 输出文件

训练完成后，将生成以下文件：

- `processed_data.pkl`: 预处理后的数据
- `checkpoints/best_model.pth`: 最佳模型权重
- `checkpoints/test_results.pkl`: 测试集结果
- `training_curves.png`: 训练曲线图

## 扩展方向

1. **距离加权图**: 使用城市间距离构建加权图
2. **移动性图**: 使用人口流动数据构建图
3. **多室模型**: 扩展到SEIR等更复杂的模型
4. **城市特定参数**: 为每个城市学习不同的SIS参数
5. **外部特征**: 加入天气、温度等外部特征

## 引用

如果使用本代码，请引用相关论文。

## 许可证

MIT License

