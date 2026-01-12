"""
可视化训练结果和预测效果
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_results(results_path='checkpoints/test_results.pkl'):
    """加载测试结果"""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_predictions_vs_targets(results, cities, num_cities_to_plot=10, save_path='predictions.png'):
    """
    绘制预测值 vs 真实值
    
    Args:
        results: 测试结果字典
        cities: 城市列表
        num_cities_to_plot: 要绘制的城市数量
        save_path: 保存路径
    """
    predictions = results['test_predictions']
    targets = results['test_targets']
    
    # 选择前N个城市
    num_cities_to_plot = min(num_cities_to_plot, len(cities))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(num_cities_to_plot):
        ax = axes[i]
        city_pred = predictions[:, i]
        city_target = targets[:, i]
        
        # 散点图
        ax.scatter(city_target, city_pred, alpha=0.5, s=10)
        
        # 对角线（完美预测）
        min_val = min(city_target.min(), city_pred.min())
        max_val = max(city_target.max(), city_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('True Cases')
        ax.set_ylabel('Predicted Cases')
        ax.set_title(f'{cities[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"预测对比图已保存到 {save_path}")

def plot_time_series(results, cities, num_samples=100, num_cities_to_plot=5, save_path='time_series.png'):
    """
    绘制时间序列预测
    
    Args:
        results: 测试结果字典
        cities: 城市列表
        num_samples: 要绘制的时间步数
        num_cities_to_plot: 要绘制的城市数量
        save_path: 保存路径
    """
    predictions = results['test_predictions']
    targets = results['test_targets']
    
    num_samples = min(num_samples, len(predictions))
    num_cities_to_plot = min(num_cities_to_plot, len(cities))
    
    fig, axes = plt.subplots(num_cities_to_plot, 1, figsize=(15, 3*num_cities_to_plot))
    if num_cities_to_plot == 1:
        axes = [axes]
    
    time_steps = np.arange(num_samples)
    
    for i in range(num_cities_to_plot):
        ax = axes[i]
        city_pred = predictions[:num_samples, i]
        city_target = targets[:num_samples, i]
        
        ax.plot(time_steps, city_target, 'b-', label='True', linewidth=1.5)
        ax.plot(time_steps, city_pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cases')
        ax.set_title(f'{cities[i]} - Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"时间序列图已保存到 {save_path}")

def plot_error_distribution(results, save_path='error_distribution.png'):
    """绘制误差分布"""
    predictions = results['test_predictions']
    targets = results['test_targets']
    
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 误差分布直方图
    axes[0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    
    # 绝对误差分布
    axes[1].hist(abs_errors.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].axvline(x=abs_errors.mean(), color='r', linestyle='--', linewidth=2, 
                    label=f'Mean: {abs_errors.mean():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"误差分布图已保存到 {save_path}")

def print_summary_statistics(results):
    """打印汇总统计信息"""
    predictions = results['test_predictions']
    targets = results['test_targets']
    
    mae = results['test_mae']
    rmse = results['test_rmse']
    
    # 计算每个城市的指标
    city_maes = np.mean(np.abs(predictions - targets), axis=0)
    city_rmses = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    
    print("\n" + "=" * 60)
    print("测试集性能汇总")
    print("=" * 60)
    print(f"总体 MAE: {mae:.4f}")
    print(f"总体 RMSE: {rmse:.4f}")
    print(f"平均绝对误差百分比: {np.mean(np.abs((predictions - targets) / (targets + 1))) * 100:.2f}%")
    print("\n各城市性能（前10名）:")
    print("-" * 60)
    
    # 按MAE排序
    sorted_indices = np.argsort(city_maes)[:10]
    for idx in sorted_indices:
        print(f"城市 {idx}: MAE={city_maes[idx]:.4f}, RMSE={city_rmses[idx]:.4f}")

def main():
    # 加载数据以获取城市列表
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    cities = data['cities']
    
    # 加载结果
    results = load_results()
    
    # 打印统计信息
    print_summary_statistics(results)
    
    # 绘制图表
    print("\n正在生成可视化图表...")
    plot_predictions_vs_targets(results, cities, num_cities_to_plot=10)
    plot_time_series(results, cities, num_samples=200, num_cities_to_plot=5)
    plot_error_distribution(results)
    
    print("\n可视化完成！")

if __name__ == "__main__":
    main()


