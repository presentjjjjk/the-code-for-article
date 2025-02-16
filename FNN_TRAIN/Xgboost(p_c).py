import numpy as np
import xgboost as xgb
import random
import os
import pickle
import json
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 设置随机种子
def set_random_seed(seed_value=42):
    """
    设置所有相关随机种子以确保结果可重复性
    Args:
        seed_value: 随机种子值
    """
    # 1. 设置Python随机种子
    random.seed(seed_value)
    
    # 2. 设置numpy随机种子
    np.random.seed(seed_value)
    
    # 4. 设置Python环境变量
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 针对GPU的设置
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
# 调用函数设置随机种子
set_random_seed(42)

# 加载数据
file_path = 'dataset_pc.csv'
data = pd.read_csv(file_path, encoding='latin1')

# 特征与目标变量（保持原特征不变）
X = data[[
    'Dipole Moment (Debye)', 
    'Polarizability',
    'free_energy',
    # part2 拓扑图形描述符
    'Chi0v',
    'Chi0n',
    'Chi1v',
    'Chi1n',
    'Chi2v',
    'Chi2n',
    'Chi3v',
    'Chi3n',
    'Chi4v',
    'Chi4n',
    'Kappa1','Kappa2',#'Kappa3',
    'HallKierAlpha',
    #'Ipc',
    'BalabanJ',
    # 分子表面积描述符
    'TPSA',
    'LabuteASA',
    'SMR_VSA1', # 具有较高负电性的分子表面积
    'SMR_VSA9', # 具有较高正电性的分子表面积
    'PEOE_VSA1',
    'PEOE_VSA14',
    # 氢键描述符
    'HBD','HBA',
    # 结构信息描述符
    'Num Rotatable Bonds',
    'Num Aromatic Atoms','Num Aromatic Rings','Num Aromatic Bonds',
    'BertzCT',
    'Volume',
    'Sphericity',
    # 静电势相关描述符
    "Min_value", 
    "Max_value",
    "Overall_surface_area",
    "Positive_surface_area",
    "Negative_surface_area",
    "Overall_average",
    "Positive_average",
    "Negative_average",
    "Overall_variance",
    "Positive_variance",
    "Negative_variance",
    "Balance",
    "Internal_charge_separation",
    "MPI",
    "Nonpolar_surface_area",
    "Polar_surface_area",
    "Overall_skewness",
    "Positive_skewness",
    "Negative_skewness",
    # 补充描述符-辛水分配指数,分子量
    'LogP',
    'Molecular Weight',
]]
y = data['p_c']



# 初始化交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

# 存储特征重要性
feature_importance_scores = []

# 交叉验证循环
best_r2 = -float('inf')
best_model = None
best_evals_result = None  # 新增：保存最佳模型的评估结果

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    # 划分开发集和测试集
    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
        X_temp, y_temp, test_index, test_size=0.5, random_state=fold
    )

    from xgboost import XGBRegressor, callback

    # 初始化模型
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_dev, y_dev)],  # 添加训练集到评估集
        early_stopping_rounds=300,   # 提前停止
        verbose=True,
        eval_metric='rmse',          # 使用RMSE作为评估指标
    )

    # 预测与评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 存储结果
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)
    
    # 收集特征重要性
    feature_importance_scores.append(model.feature_importances_)
    
    # 保存最佳模型
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_evals_result = model.evals_result()  # 保存评估结果

# 输出统计结果
print(f"Best R²: {best_r2:.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average MRE: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

# 特征重要性分析
mean_importance = np.mean(feature_importance_scores, axis=0)
std_importance = np.std(feature_importance_scores, axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_importance,
    'Std': std_importance
}).sort_values('Importance', ascending=False)

import seaborn as sns


sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

# 使用渐变色
cmap = plt.cm.Blues
colors = [cmap(i / 10) for i in range(10)]  # 只需要10个颜色

bars = plt.barh(range(10), feature_importance_df['Importance'].values[:10], 
                xerr=feature_importance_df['Std'].values[:10], align='center',
                color=colors, edgecolor='black', ecolor='gray', capsize=5, error_kw={'elinewidth':1.5})

plt.gca().invert_yaxis()
plt.yticks(range(10), feature_importance_df['Feature'].values[:10], fontsize=10)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('Top 10 Feature Importance with Standard Deviation', fontsize=14)

# 添加数值标注，增加偏移量
for i, bar in enumerate(bars):
    # 增加偏移量，确保数值标注不与误差棒重合
    offset = bar.get_width() * 0.1  # 根据bar的宽度动态调整偏移量
    plt.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, 
             f'{bar.get_width():.2f}', va='center', fontsize=20, color='black')

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 新增：绘制最佳模型的训练曲线
if best_evals_result is not None:
    plt.figure(figsize=(10, 6))
    
    # 提取训练和验证的RMSE值
    train_rmse = best_evals_result['validation_0']['rmse']
    val_rmse = best_evals_result['validation_1']['rmse']
        
    # 绘制原始曲线
    plt.plot(train_rmse, label='Training RMSE', alpha=1, color='tab:blue')
    plt.plot(val_rmse, label='Validation RMSE', alpha=1, color='tab:orange')
    
    # 标记最佳迭代次数
    best_iter = best_model.best_iteration
    plt.axvline(best_iter, color='gray', linestyle='--', 
                label=f'Best Iteration ({best_iter})')
    
    # 图表装饰
    plt.xlabel('Boosting Rounds', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Best Model Training Progress with Early Stopping', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No training curves available.")




