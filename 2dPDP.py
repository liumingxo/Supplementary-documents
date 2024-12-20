import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump, load
import tkinter as tk
from tkinter import ttk

# 设置字体为支持中文的字体（请根据你的系统环境调整）
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 设置全局样式
sns.set(style="whitegrid")

# 加载数据
data = pd.read_excel('D:\\Google\\sj_augmented.xlsx')

# 定义输入和输出列
input_columns = ['C', 'H', 'O', 'N', 'K', 'P', 'M', 'Ash', 'FC', 'LT', 'T', 'TR']
output_columns = ['BY', 'C', 'H', 'O', 'N', 'P', 'K']

# 分离输入和输出数据
X = data[input_columns]
y = data[output_columns]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 转成带列名的DataFrame，方便后续操作
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=input_columns)
X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=input_columns)

# 初始化XGBoost模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1
)

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse, cv_rmse, cv_r2 = [], [], []
all_y_test_cv, all_y_pred_cv = [], []

for train_index, test_index in kf.split(X_train_scaled_df):
    X_train_cv, X_test_cv = X_train_scaled_df.iloc[train_index], X_train_scaled_df.iloc[test_index]
    y_train_cv, y_test_cv = y_train_scaled[train_index], y_train_scaled[test_index]

    model.fit(X_train_cv, y_train_cv)
    y_pred_cv = model.predict(X_test_cv)

    # 反标准化
    y_pred_cv = y_scaler.inverse_transform(y_pred_cv)
    y_test_cv = y_scaler.inverse_transform(y_test_cv)

    all_y_test_cv.extend(y_test_cv)
    all_y_pred_cv.extend(y_pred_cv)

    # 计算性能指标（对多输出取平均或逐列评估都可以，这里示范整体平均）
    mse = mean_squared_error(y_test_cv, y_pred_cv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_cv, y_pred_cv, multioutput='uniform_average')

    cv_mse.append(mse)
    cv_rmse.append(rmse)
    cv_r2.append(r2)

all_y_test_cv = np.array(all_y_test_cv)
all_y_pred_cv = np.array(all_y_pred_cv)

print(f'交叉验证平均MSE: {np.mean(cv_mse)}')
print(f'交叉验证平均RMSE: {np.mean(cv_rmse)}')
print(f'交叉验证平均R^2: {np.mean(cv_r2)}')

# 保存模型为UBJ格式
model.save_model('model.ubj')

# 可视化真实值与预测值
plt.figure(figsize=(8, 7))
plt.scatter(all_y_test_cv, all_y_pred_cv, color='#2A9D8F', alpha=0.6, label='预测值', edgecolors='w', s=28)

# 绘制理想对角线。注意：y_test与y_pred是多列，这里可视化一般只做单列或做散点矩阵
# 为简单起见，这里用所有列混合后点图示范
min_val = min(all_y_test_cv.min(), all_y_pred_cv.min())
max_val = max(all_y_test_cv.max(), all_y_pred_cv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=0.9, color='#264653', label='理想对角线')

plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值与预测值的拟合度（多输出混合可视化示意）')
plt.grid(False)
plt.tight_layout()
plt.savefig('xgbthou.eps', format='eps')
plt.show()

# ---------------------
# 以下是 2D Partial Dependence Plot (PDP) 的示例
# ---------------------

# 选择想要分析的两个输入特征索引（也可以直接用列名）
feature1_idx = 3  # 对应 input_columns[0] = 'C'
feature2_idx = 10  # 对应 input_columns[1] = 'H'

# 指定想要分析的输出列索引（在 output_columns 里），比如我们这里对第 0 列 'BY' 做分析
target_output_idx = 4

# 为了计算PDP，需要在特征1和特征2的取值区间上建立网格
x1_range = np.linspace(X_train_scaled_df.iloc[:, feature1_idx].min(),
                       X_train_scaled_df.iloc[:, feature1_idx].max(),
                       50)
x2_range = np.linspace(X_train_scaled_df.iloc[:, feature2_idx].min(),
                       X_train_scaled_df.iloc[:, feature2_idx].max(),
                       50)

X1, X2 = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[X1.ravel(), X2.ravel()]  # shape: (2500, 2)

# 其余特征用均值填充
other_features = X_train_scaled_df.drop(X_train_scaled_df.columns[[feature1_idx, feature2_idx]], axis=1)
mean_values = other_features.mean().values[np.newaxis, :]  # shape: (1, n_features-2)
grid_points_expanded = np.repeat(mean_values, len(grid_points), axis=0)  # shape: (2500, n_features-2)

# 合并网格和均值填充的列，得到 shape: (2500, n_features)
X_grid = np.hstack([grid_points, grid_points_expanded])

# 模型预测
y_pred_grid = model.predict(X_grid)  # shape: (2500, 7)  对于多输出会有7列

# 只对 target_output_idx 这一列做 PDP
y_pred_grid_single = y_pred_grid[:, target_output_idx]  # shape: (2500,)

# 将预测结果重塑为网格形状 (50, 50)
y_pred_grid_single = y_pred_grid_single.reshape(X1.shape)

# 绘制2D PDP图
plt.figure(figsize=(8, 6))
cp = plt.contourf(X1, X2, y_pred_grid_single, levels=20, cmap='viridis')
plt.colorbar(cp)

plt.xlabel(input_columns[feature1_idx])
plt.ylabel(input_columns[feature2_idx])
plt.title(f'2D PDP for output: {output_columns[target_output_idx]}')
plt.tight_layout()

plt.savefig(f'2d_pdp_{output_columns[target_output_idx]}.png', dpi=300)
plt.show()



