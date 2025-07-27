import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
# from scipy.signal import savgol_filter
# font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 宋体字体路径
# font_chinese = FontProperties(fname=font_path,size=18)
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置设备为GPU（如果可用）
device = torch.device('cpu')
print(torch.cuda.is_available())
# 读取数据
datasets = pd.read_excel(r"D:\datasets\data0545.xlsx", header=0) #读你自己的文件
datasets.columns = ["surge","sway","heave","roll","pitch","yaw","fairlead"]
# 数据截断和缩放
truncation = 8000
lookback = 8#需要时间以来在这里设置
dims = ["surge","sway","heave","roll","pitch","yaw","fairlead"]
data_list = datasets[dims].iloc[truncation:].to_numpy()

# 归一化数据
scalers = [MinMaxScaler() for _ in dims]
data_scaled = np.column_stack([scaler.fit_transform(data_list[:, i].reshape(-1, 1)).ravel() for i, scaler in enumerate(scalers)])

def create_dataset(data_scaled, lookback):
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i + lookback, :-1])
        y.append(data_scaled[i + lookback, -1])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, lookback)
print(X.shape, y.shape)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch 张量
# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)
y_data = data_list[:,-1]
split_index = int(0.8 * len(X))
data_train, data_test = y_data[:split_index], y_data[split_index:]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_prob = 0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.linear(x)  # 取最后一个时间步的输出
        return x
# 实例化模型
model = LSTMModel(input_size=6, hidden_size=128, output_size=1, num_layers=3, dropout_prob=0.1).to(device)
# criterion = nn.MSELoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # 加入L2正则化
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# # 早停机制
# class EarlyStopping:
#     def __init__(self, patience=10, delta=0):
#         self.patience = patience  # 容忍的验证损失不改善的轮数
#         self.delta = delta  # 最小改善阈值
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
# # 初始化早停机制
# early_stopping = EarlyStopping(patience=5, delta=0.001)
# # 训练模型
# train_dataset = TensorDataset(X_train, y_train)
# loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# n_epochs = 35
# best_loss = float('inf')
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     #更新学习率
#     # scheduler.step()
#     # 验证
#     if epoch % 5 == 0:
#         model.eval()
#         with torch.no_grad():
#             y_pred_train = model(X_train)
#             train_rmse = np.sqrt(criterion(y_pred_train.cpu(), y_train.cpu()))
#             y_pred_test = model(X_test)
#             test_rmse = np.sqrt(criterion(y_pred_test.cpu(), y_test.cpu()))
#         print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Loss = {loss:.4f}")
#
#         # 早停机制检查
#         early_stopping(test_rmse)
#         if early_stopping.early_stop:
#             print("Early stopping triggered!",epoch)
#             break
# # # 保存最佳模型权重
# torch.save(model.state_dict(), 'lstm_model_weights-0545.pth')
model.load_state_dict(torch.load('lstm_model_weights-0545.pth'))
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.squeeze()
    y_pred_inv = scalers[-1].inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).ravel()

y_test_original = data_list[:, -1][split_index:]
y_pred_inv = y_pred_inv/1000
y_test_original= y_test_original/1000

# file_path = "output952.xlsx"  # 这一部分是将数据输出
# df = pd.read_excel(file_path)
# df.insert(7, 'LSTM4256', y_pred_inv)  # 在第二列（索引为1的位置）插入数据
# # # 将修改后的 DataFrame 写回 Excel 文件（覆盖原文件）
# df.to_excel(file_path, index=False)  # index=False 表示不保存索引列
# Calculate RMSE and MAPE
test_rmse_value = np.sqrt(np.mean((y_pred_inv - y_test_original[lookback:]) ** 2))
mae_value = np.mean(np.abs(y_pred_inv - y_test_original[lookback:]))
r2_value = r2_score(y_test_original[lookback:], y_pred_inv)
#
test_mape_value = np.mean(np.abs((y_pred_inv - y_test_original[lookback:]) / y_test_original[lookback:])) * 100

x = np.arange(0, len(y_pred_inv) * 0.05, 0.05) #0.2是我的时间间隔
x = x[1:]

# file_path = "output952.xlsx"  # 这一部分是将数据输出
# df = pd.read_excel(file_path)
# df.insert(12, 'LSTM', y_pred_inv)  # 在第二列（索引为1的位置）插入数据
# # # 将修改后的 DataFrame 写回 Excel 文件（覆盖原文件）
# df.to_excel(file_path, index=False)  # index=False 表示不保存索引列

print(test_rmse_value,r2_value,mae_value, test_mape_value)
# Plot results with enhanced aesthetics
plt.figure(figsize=(12, 10.5))
plt.ylim(700,3500)
ticks = np.arange(700, 3600, 400)  # 注意：arange 的结束值是开区间，所以要加 1
plt.yticks(ticks)
plt.tick_params(axis='both', which='major', labelsize=24)
# plt.plot(x_new,y_test_original[lookback:], label='Original Data', color='blue', linewidth=1)
# plt.plot(x_new,y_pred_inv, label='Predicted Data (CNN+LSTM+Attention)', color='red', linestyle='--', linewidth=1.8)
plt.plot(x,y_pred_inv, label='Predicted Value (LSTM)',color='#FF5733', linestyle='--', linewidth=2)
plt.plot(x,y_test_original[lookback:], label='Actual Value', color='#2E86C1', linewidth=2)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Fairlead Tension (kN)', fontsize=24)
plt.legend()
plt.text(0.2, 0.92, f'R²: {r2_value:.3f}', transform=plt.gca().transAxes, fontsize=26,
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
#
plt.show()

