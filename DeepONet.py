import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 自定义数据集类
class WindWaveMotionDataset(Dataset):
    def __init__(self, wind_file, wave_file, motion_file):
        # 读取数据
        self.wind_data = pd.read_excel(wind_file).to_numpy()
        self.wave_data = pd.read_excel(wave_file).to_numpy()
        self.motion_data = pd.read_excel(motion_file).to_numpy()

        # 确保数据维度一致
        assert len(self.wind_data) == len(self.wave_data) == len(self.motion_data), "数据长度不一致"

    def __len__(self):
        return len(self.wind_data)

    def __getitem__(self, idx):
        wind = self.wind_data[idx]  # 风速时间序列
        wave = self.wave_data[idx]  # 波浪时间序列
        motion = self.motion_data[idx]  # 浮式风机的运动响应
        return torch.tensor(wind, dtype=torch.float32), torch.tensor(wave, dtype=torch.float32), torch.tensor(motion, dtype=torch.float32)

# 数据集文件路径
wind_file = "wind.xlsx"
wave_file = "wave.xlsx"
motion_file = "motion.xlsx"

# 创建数据集和数据加载器
dataset = WindWaveMotionDataset(wind_file, wave_file, motion_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class DeepONet(nn.Module):
    def __init__(self, branch_input_size, trunk_input_size, hidden_size=64):
        super(DeepONet, self).__init__()

        # 分支网络：处理风速和波浪输入
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 主干网络：处理时间点输入
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 最终全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, branch_input, trunk_input):
        branch_out = self.branch_net(branch_input)  # 分支网络输出
        trunk_out = self.trunk_net(trunk_input)  # 主干网络输出
        combined = branch_out * trunk_out  # 结合分支和主干输出
        output = self.fc(combined)
        return output


# 模型实例化
branch_input_size = 2  # 风速和波浪
trunk_input_size = 1  # 时间点
model = DeepONet(branch_input_size, trunk_input_size, hidden_size=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 模型训练
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for wind, wave, motion in dataloader:
        # 构造分支输入和主干输入
        branch_input = torch.cat((wind, wave), dim=1)  # 合并风速和波浪
        trunk_input = torch.linspace(0, 1, motion.size(1)).unsqueeze(0).repeat(motion.size(0), 1)  # 时间点
        trunk_input = trunk_input.unsqueeze(2)  # 添加一个维度以适配网络

        # 目标输出
        target = motion

        # 前向传播
        output = model(branch_input, trunk_input)
        loss = criterion(output.squeeze(), target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    for wind, wave, motion in dataloader:
        branch_input = torch.cat((wind, wave), dim=1)
        trunk_input = torch.linspace(0, 1, motion.size(1)).unsqueeze(0).repeat(motion.size(0), 1)
        trunk_input = trunk_input.unsqueeze(2)

        prediction = model(branch_input, trunk_input).squeeze().numpy()
        true_response = motion.numpy()
        break  # 仅测试一个批次

# 可视化结果
time_points = np.linspace(0, 1, true_response.shape[1])
plt.figure(figsize=(10, 6))
plt.plot(time_points, true_response[0], label="True Response")
plt.plot(time_points, prediction[0], label="Predicted Response", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Motion Response")
plt.legend()
plt.show()
