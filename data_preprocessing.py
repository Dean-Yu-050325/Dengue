"""
数据预处理模块：将原始登革热数据转换为按城市和日期聚合的格式
"""
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import pickle

def load_and_preprocess_data(csv_path, use_notification_date=True):
    """
    加载并预处理登革热数据
    
    Args:
        csv_path: CSV文件路径
        use_notification_date: 如果True，使用Date_Notification；否则使用Date_Onset
    
    Returns:
        aggregated_data: DataFrame，包含日期、城市、病例数
        city_list: 城市列表（排序后）
        date_range: 日期范围
    """
    print("正在加载数据...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 选择日期列
    if use_notification_date:
        date_col = 'Date_Notification'
    else:
        date_col = 'Date_Onset'
    
    # 选择城市列（使用MOI_County_living作为标准城市名称）
    city_col = 'MOI_County_living'
    
    # 过滤有效数据
    df = df[df[date_col].notna()]
    df = df[df[city_col].notna()]
    
    # 转换日期格式
    df[date_col] = pd.to_datetime(df[date_col], format='%Y/%m/%d', errors='coerce')
    df = df[df[date_col].notna()]
    
    # 获取病例数（通常是1，但可能有多个）
    df['cases'] = df['Number_of_confirmed_cases'].fillna(1).astype(int)
    
    # 按日期和城市聚合
    print("正在聚合数据...")
    aggregated = df.groupby([date_col, city_col])['cases'].sum().reset_index()
    aggregated.columns = ['date', 'city', 'cases']
    
    # 获取所有城市和日期
    all_cities = sorted(aggregated['city'].unique())
    all_dates = pd.date_range(
        start=aggregated['date'].min(),
        end=aggregated['date'].max(),
        freq='D'
    )
    
    # 创建完整的时间序列矩阵
    print("正在创建时间序列矩阵...")
    date_city_matrix = pd.DataFrame(
        index=all_dates,
        columns=all_cities,
        data=0
    )
    
    # 填充数据
    for _, row in aggregated.iterrows():
        date = row['date']
        city = row['city']
        cases = row['cases']
        if date in date_city_matrix.index and city in date_city_matrix.columns:
            date_city_matrix.loc[date, city] += cases
    
    # 转换为numpy数组
    data_matrix = date_city_matrix.values.astype(np.float32)
    
    print(f"数据形状: {data_matrix.shape}")
    print(f"日期范围: {all_dates[0]} 到 {all_dates[-1]}")
    print(f"城市数量: {len(all_cities)}")
    print(f"总病例数: {data_matrix.sum():.0f}")
    
    return {
        'data_matrix': data_matrix,
        'cities': all_cities,
        'dates': all_dates,
        'date_city_df': date_city_matrix
    }

def create_sequences(data_matrix, window_size=14, forecast_horizon=1):
    """
    创建时间序列样本
    
    Args:
        data_matrix: 形状为 (T, N) 的数组，T是时间步数，N是城市数
        window_size: 输入窗口大小
        forecast_horizon: 预测步数（未来多少天）
    
    Returns:
        X: 输入序列，形状 (samples, window_size, N)
        y: 目标值，形状 (samples, N)
    """
    T, N = data_matrix.shape
    samples = T - window_size - forecast_horizon + 1
    
    X = np.zeros((samples, window_size, N), dtype=np.float32)
    y = np.zeros((samples, N), dtype=np.float32)
    
    for i in range(samples):
        X[i] = data_matrix[i:i+window_size]
        y[i] = data_matrix[i+window_size+forecast_horizon-1]
    
    return X, y

def split_train_test(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    划分训练集、验证集和测试集
    
    Args:
        X: 输入数据
        y: 目标数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_X, train_y, val_X, val_y, test_X, test_y
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_X = X[:n_train]
    train_y = y[:n_train]
    val_X = X[n_train:n_train+n_val]
    val_y = y[n_train:n_train+n_val]
    test_X = X[n_train+n_val:]
    test_y = y[n_train+n_val:]
    
    return train_X, train_y, val_X, val_y, test_X, test_y

if __name__ == "__main__":
    # 预处理数据
    data = load_and_preprocess_data('Dengue_Daily_EN.csv', use_notification_date=True)
    
    # 创建序列
    window_size = 14
    X, y = create_sequences(data['data_matrix'], window_size=window_size)
    
    # 划分数据集
    train_X, train_y, val_X, val_y, test_X, test_y = split_train_test(X, y)
    
    # 保存预处理后的数据
    processed_data = {
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y,
        'test_X': test_X,
        'test_y': test_y,
        'cities': data['cities'],
        'dates': data['dates'],
        'window_size': window_size
    }
    
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n数据预处理完成！")
    print(f"训练集: {train_X.shape}")
    print(f"验证集: {val_X.shape}")
    print(f"测试集: {test_X.shape}")
    print(f"数据已保存到 processed_data.pkl")

