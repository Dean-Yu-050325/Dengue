"""
完整的训练管道：从数据预处理到模型训练
"""
import os
import sys
import subprocess

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        sys.exit(1)
    else:
        print(result.stdout)
        return True

def main():
    print("=" * 60)
    print("Epidemiology-Informed Spatio-Temporal GNN Pipeline")
    print("=" * 60)
    
    # 步骤1: 数据预处理
    if not os.path.exists('processed_data.pkl'):
        print("\n步骤 1/3: 数据预处理")
        run_command("python data_preprocessing.py", "正在预处理数据...")
    else:
        print("\n步骤 1/3: 数据预处理 (已存在，跳过)")
    
    # 步骤2: 训练模型
    print("\n步骤 2/3: 训练模型")
    run_command("python train.py", "正在训练模型...")
    
    # 步骤3: 可视化结果
    if os.path.exists('checkpoints/test_results.pkl'):
        print("\n步骤 3/3: 可视化结果")
        run_command("python visualize_results.py", "正在生成可视化...")
    else:
        print("\n步骤 3/3: 可视化结果 (测试结果不存在，跳过)")
    
    print("\n" + "=" * 60)
    print("管道执行完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - processed_data.pkl: 预处理后的数据")
    print("  - checkpoints/best_model.pth: 最佳模型权重")
    print("  - checkpoints/test_results.pkl: 测试结果")
    print("  - training_curves.png: 训练曲线")
    print("  - predictions.png: 预测对比图")
    print("  - time_series.png: 时间序列图")
    print("  - error_distribution.png: 误差分布图")

if __name__ == "__main__":
    main()


