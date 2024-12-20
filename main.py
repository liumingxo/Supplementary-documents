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
# 设置字体为支持中文的字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 请根据你的系统调整路径
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

# 初始化XGBoost模型，移除早停法参数
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

# 准备5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储交叉验证结果
cv_mse = []
cv_rmse = []
cv_r2 = []
all_y_test_cv = []  # 用于收集所有测试集的真实值
all_y_pred_cv = []  # 用于收集所有测试集的预测值

# 执行交叉验证
for train_index, test_index in kf.split(X_train_scaled):
    X_train_cv, X_test_cv = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_cv, y_test_cv = y_train_scaled[train_index], y_train_scaled[test_index]

    # 训练模型
    model.fit(X_train_cv, y_train_cv)

    # 进行预测
    y_pred_cv = model.predict(X_test_cv)

    # 反标准化预测值
    y_pred_cv = y_scaler.inverse_transform(y_pred_cv)

    # 反标准化真实值
    y_test_cv = y_scaler.inverse_transform(y_test_cv)
    all_y_test_cv.extend(y_test_cv)
    all_y_pred_cv.extend(y_pred_cv)

    # 计算性能指标
    mse = mean_squared_error(y_test_cv, y_pred_cv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_cv, y_pred_cv)

    # 存储结果
    cv_mse.append(mse)
    cv_rmse.append(rmse)
    cv_r2.append(r2)

all_y_pred_cv = np.array(all_y_pred_cv)

# 计算平均性能指标
average_mse = np.mean(cv_mse)
average_rmse = np.mean(cv_rmse)
average_r2 = np.mean(cv_r2)

# 打印交叉验证的平均性能指标
print(f'交叉验证平均MSE: {average_mse}')
print(f'交叉验证平均RMSE: {average_rmse}')
print(f'交叉验证平均R^2: {average_r2}')

# 在这里保存模型为JSON格式
model.save_model('model.ubj')

# 可视化真实值与预测值的对比
plt.figure(figsize=(8, 7))
plt.scatter(all_y_test_cv, all_y_pred_cv, color='#2A9D8F',alpha=0.6, label='预测值',edgecolors='w',s=28)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k-', lw=0.9,color='#264653', label='真实值')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值与预测值的拟合度')
plt.grid(False)
plt.tight_layout()

plt.savefig('xgbthou.eps', format='eps')
plt.show()

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# 选择特征索引
features = [4, 5]  # 只选择两个输入特征

# 创建子图
fig, axes = plt.subplots(1, len(features), figsize=(12, 6))  # 调整顶层应该适尺寸

# 设置Matplotlib样式
plt.style.use('default')  # 改回默认样式

# 迭代每个特征，绘制PDP和ICE
for i, feature in enumerate(features):
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_test_scaled,
        [feature],
        ax=axes[i],
        kind="both",
        target=6  # 添加target参数，选择第一个目标变量
    )

    # 设置标题
    axes[i].set_title(f'Feature: {input_columns[feature]}', fontsize=14)

    # 去除网格线
    axes[i].grid(False)

    # 去除图像背景
    axes[i].patch.set_facecolor('white')

    # 设置小标符的样式
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_visible(True)
    axes[i].spines['bottom'].set_visible(True)

    # 设置x和y轴标签
    axes[i].set_xlabel(input_columns[feature], fontsize=12)
    axes[i].set_ylabel('Partial Dependence', fontsize=12)

    # 设置更大的刻度字体
    axes[i].tick_params(axis='both', which='major', labelsize=10)
# 增加整体布局调整
plt.tight_layout()
# 保存PDP和ICE分析结果为PDF
plt.savefig('pdp_ice_analysis_modified4.pdf', format='pdf', dpi=600, bbox_inches='tight')
# 显示图像
plt.show()


print(all_y_test_cv, all_y_pred_cv)
# 创建一个空的DataFrame来存储真实值和预测值
results_df = pd.DataFrame()

# 将真实值和预测值添加到DataFrame
for i, col in enumerate(output_columns):
    results_df[f'{col}_真实值'] = y_test_cv[:, i]
    results_df[f'{col}_预测值'] = y_pred_cv[:, i]

# 尝试将结果保存到Excel文件
try:
    results_df.to_excel('预测结果22.xlsx', index=False)
    print('预测结果已成功保存到Excel文件。')
except Exception as e:
    print(f'保存Excel文件时出现错误：{e}')
#
# SHAP分析和Tkinter GUI代码
# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# 如果输出是单一的，则将其转换为二维数组
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    shap_values = [shap_values]

# 初始化一个空的 DataFrame 来存储所有输出参数的 SHAP 值
shap_df_mean_all = pd.DataFrame()

# 对于每个输出参数，计算其 SHAP 值的平均绝对值
for i, shap_values_output in enumerate(shap_values):
    # 将 SHAP 值转换为 DataFrame
    shap_df = pd.DataFrame(shap_values_output, columns=input_columns)

    # 计算特征平均绝对 SHAP 值
    shap_df_mean = pd.DataFrame({
        'Feature': input_columns,
        f'Mean |SHAP Value| ({output_columns[i]})': np.abs(shap_df).mean().values
    })

    # 将结果合并到总的 DataFrame 中
    if shap_df_mean_all.empty:
        shap_df_mean_all = shap_df_mean
    else:
        shap_df_mean_all = pd.merge(shap_df_mean_all, shap_df_mean, on='Feature')

# 将特征平均绝对 SHAP 值排序
shap_df_mean_all = shap_df_mean_all.set_index('Feature')
shap_df_mean_sorted = shap_df_mean_all.mean(axis=1).sort_values(ascending=True).index
shap_df_mean_all = shap_df_mean_all.loc[shap_df_mean_sorted]

# 前三个输出的 SHAP 值
shap_df_mean_first_three = shap_df_mean_all.iloc[:, :3]

# 后四个输出的 SHAP 值
shap_df_mean_last_four = shap_df_mean_all.iloc[:, 3:]

# 绘制前三个输出参数的 SHAP 值条形图
shap_df_mean_first_three.plot(kind='barh', figsize=(12, 10), cmap='viridis')
plt.title('Mean |SHAP Values| per Feature for Outputs 1-3')
plt.xlabel('Mean |SHAP Value|')
plt.ylabel('Feature')
plt.grid(False)
plt.legend(title='Output Parameters')
plt.tight_layout()
plt.savefig('first_three_outputs_shap.png', dpi=600)
plt.show()

# 绘制后四个输出参数的 SHAP 值条形图
shap_df_mean_last_four.plot(kind='barh', figsize=(12, 10), cmap='viridis')
plt.title('Mean |SHAP Values| per Feature for Outputs 4-7')
plt.xlabel('Mean |SHAP Value|')
plt.ylabel('Feature')
plt.grid(False)
plt.legend(title='Output Parameters')
plt.tight_layout()
plt.savefig('last_four_outputs_shap.png', dpi=600)
plt.show()
# 将所有 SHAP 值保存在 Excel 文件中
shap_df_mean_all.to_excel('shap_values.xlsx')
# 打印出 SHAP 值
print(shap_df_mean_all)
# 创建 Explanation 对象
sample_index = 0  # 选择一个样本
shap_explanation = shap.Explanation(
    values=shap_values[0][sample_index],  # 选择对应输出的 SHAP 值
    base_values=explainer.expected_value[0],  # 使用对应的基线值
    data=X_test.iloc[sample_index],  # 相应的输入数据
    feature_names=input_columns
)

# 生成第一个输出的瀑布图
plt.figure(figsize=(30, 15))
shap.plots.waterfall(shap_explanation, max_display=15, show=False)
ax = plt.gca()
ax.grid(False)  # 去掉网格线
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.savefig('opygq4.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# 为第一个输出生成蜂群图汇总图，并将颜色设置为类似于上传的图像
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[6], X_test, plot_type="dot", feature_names=input_columns,show=False)
ax = plt.gca()
ax.grid(False)  # 去掉网格线
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.savefig('dot7.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# 主窗口
root = tk.Tk()
root.title("XGBoost Prediction")
root.geometry('1000x700')  # 调整窗口大小
root.configure(bg='#f0f4f7')  # 背景色

# 创建标题标签
title_label = tk.Label(root, text="Biochar Properties Prediction", font=("Arial", 18, "bold"), bg='#f0f4f7', fg='#333333')
title_label.pack(pady=20)

# 创建输入和输出Frame
frame_input = ttk.Frame(root, padding="10", relief="solid", borderwidth=1)
frame_input.place(relx=0.05, rely=0.2, relwidth=0.4, relheight=0.7)

frame_output = ttk.Frame(root, padding="10", relief="solid", borderwidth=1)
frame_output.place(relx=0.55, rely=0.2, relwidth=0.4, relheight=0.7)

# 输入字段标题
tk.Label(frame_input, text="Input Data", font=("Arial", 14, "bold"), bg="#ffffff", fg="#333333").pack(pady=10)

# 模拟输入和输出列
input_columns = ["C", "H", "O", "N", "K", "P", "M", "Ash", "FC", "LT", "T", "TR"]
output_columns = ["BY", "C", "H", "O", "N", "P", "K"]
entries = {}

# 使用 grid 布局输入字段
content_frame = ttk.Frame(frame_input)
content_frame.pack(fill="both", expand=True)

# 对列进行权重配置，使得各列在窗口大小变化时可以自动调整
for i in range(4):
    content_frame.columnconfigure(i, weight=1)

# 将12个字段分为左右两组，每组6个字段
left_columns = input_columns[:6]
right_columns = input_columns[6:]

# 在Grid中布置左侧字段(0,1列)和右侧字段(2,3列)
for i, col_name in enumerate(left_columns):
    lbl = ttk.Label(content_frame, text=col_name, font=("Arial", 12))
    ent = ttk.Entry(content_frame, font=("Arial", 12), width=15)
    lbl.grid(row=i, column=0, padx=5, pady=5, sticky="E")
    ent.grid(row=i, column=1, padx=5, pady=5, sticky="W")
    entries[col_name] = ent

for i, col_name in enumerate(right_columns):
    lbl = ttk.Label(content_frame, text=col_name, font=("Arial", 12))
    ent = ttk.Entry(content_frame, font=("Arial", 12), width=15)
    lbl.grid(row=i, column=2, padx=5, pady=5, sticky="E")
    ent.grid(row=i, column=3, padx=5, pady=5, sticky="W")
    entries[col_name] = ent

# 预测按钮
predict_button = ttk.Button(root, text="Predict")
predict_button.place(relx=0.5, rely=0.92, anchor="center")

# 输出标题
tk.Label(frame_output, text="Prediction Results", font=("Arial", 14, "bold"), bg="#ffffff", fg="#333333").pack(pady=10)

# 预测函数
def predict():
    try:
        input_data = np.array([[float(entries[col].get()) for col in entries]]).astype('float32')
        prediction = np.random.rand(1, len(output_columns))  # 模拟预测输出

        # 清空输出Frame中的内容（保留标题）
        for widget in frame_output.winfo_children():
            if widget.winfo_class() == 'Label' and "Prediction Results" not in widget.cget("text"):
                widget.destroy()

        for i, col in enumerate(output_columns):
            ttk.Label(frame_output, text=f'{col}: {prediction[0][i]:.3f}', font=("Arial", 12), background="#ffffff", foreground="#444444").pack(pady=5)
    except ValueError:
        ttk.Label(frame_output, text="Invalid input. Please enter numeric values.", font=("Arial", 12), background="#FFCDD2", foreground="#000000").pack(pady=5)

predict_button.config(command=predict)

root.mainloop()
