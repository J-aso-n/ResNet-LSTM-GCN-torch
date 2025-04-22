import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from metrics import evaluate_performance
from load_data import Get_All_Data
import numpy as np
import os
import time
from tqdm import tqdm

# 之前已经实现过的残差模块和注意力模块
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(Unit, self).__init__()
        self.pool = pool
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 下采样池化
        if pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 残差支路（通道不同时或下采样时）
        if pool or in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if pool else 1)
        else:
            self.res_conv = None

        # 主干网络
        self.bn1 = nn.BatchNorm2d(in_channels if not pool else in_channels)  # 注意：使用x原始的通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x

        if self.pool:
            x = self.pool_layer(x)
            if self.res_conv is not None:
                res = self.res_conv(res)
                # 确保 res 和 x 的大小一致
                if res.shape[-2:] != x.shape[-2:]:
                    res = F.interpolate(res, size=x.shape[-2:], mode='nearest')
        elif self.res_conv is not None:
            res = self.res_conv(res)
            if res.shape[-2:] != x.shape[-2:]:
                res = F.interpolate(res, size=x.shape[-2:], mode='nearest')

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # 这里 out 和 res 形状一定一致了
        out += res
        return out

class Attention3DBlock(nn.Module):
    def __init__(self, timesteps):
        super(Attention3DBlock, self).__init__()
        self.linear = nn.Linear(timesteps, timesteps)

    def forward(self, inputs):
        # inputs: (batch, timesteps, features)
        x = inputs.permute(0, 2, 1)  # (batch, features, timesteps)
        a = self.linear(x)           # attention score
        a_probs = a.permute(0, 2, 1) # (batch, timesteps, features)
        return inputs * a_probs

class MultiInputModel(nn.Module):
    def __init__(self, time_lag):
        super(MultiInputModel, self).__init__()
        self.time_lag = time_lag

        # 输入形状: (B, 276, time_lag-1, C)
        self.conv_blocks = nn.ModuleList()
        for in_channels in [3, 3, 1]:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    Unit(32, 32),
                    Unit(32, 64, pool=True),  # 假设 pool=True 表示 MaxPool2d(kernel=2)
                    nn.Flatten()
                )
            )

        # ===== 用虚拟输入动态计算 flatten 之后的大小 =====
        dummy_input = torch.zeros(1, 3, 276, time_lag - 1)  # B, C, H, W
        with torch.no_grad():
            dummy_out = self.conv_blocks[0](dummy_input)
            flatten_dim = dummy_out.shape[1]  # e.g. 17664

        # 注意：展平后的输出维度根据实际输入计算
        self.dense_blocks = nn.ModuleList([
            nn.Linear(flatten_dim, 276) for _ in range(3)
        ])

        # input4: (B, 11, time_lag-1, 1) -> Flatten -> Dense -> LSTM -> LSTM -> Dense
        self.fc4 = nn.Linear(11 * (time_lag - 1), 276)
        self.lstm4_1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm4_2 = nn.LSTM(input_size=128, hidden_size=276, batch_first=True)
        self.fc4_out = nn.Linear(276, 276)

        # LSTM + Attention + Final Dense
        self.lstm_final = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.attention = Attention3DBlock(timesteps=276)
        self.flatten = nn.Flatten()
        self.final_dense = nn.Linear(276 * 128, 276)

    def forward(self, input1, input2, input3, input4):
        # permute -> (B, C, H, W)
        # print("1:",input1.shape)
        x1 = self.conv_blocks[0](input1.permute(0, 3, 1, 2))
        x1 = self.dense_blocks[0](x1)
        # print("1:",x1.shape)

        # print("2:",input2.shape)
        x2 = self.conv_blocks[1](input2.permute(0, 3, 1, 2))
        x2 = self.dense_blocks[1](x2)
        # print("2:",x2.shape)

        # print("3:",input3.shape)
        x3 = self.conv_blocks[2](input3.permute(0, 3, 1, 2))
        x3 = self.dense_blocks[2](x3)
        # print("3:",x3.shape)

        # print("4:",input4.shape)
        # input4: (B, 11, T-1, 1) -> Flatten -> Dense -> unsqueeze -> LSTM
        b, c, t, _ = input4.shape
        x4 = input4.view(b, -1)        # (B, 11*(T-1))
        x4 = self.fc4(x4).unsqueeze(-1)  # (B, 276, 1)
        x4, _ = self.lstm4_1(x4)              # (B, 276, 128)
        x4, _ = self.lstm4_2(x4)              # (B, 276, 276)
        x4 = x4[:, -1, :]                     # (B, 276)
        x4 = self.fc4_out(x4)                 # (B, 276)
        # print("4:",x4.shape)

        # 加和四个分支
        out = x1 + x2 + x3 + x4  # (B, 276)

        # LSTM + Attention
        out = out.unsqueeze(-1)  # (B, 276, 1)
        out, _ = self.lstm_final(out)  # (B, 276, 128)
        out = self.attention(out)      # (B, 276, 128)
        out = self.flatten(out)        # (B, 276*128)
        out = self.final_dense(out)    # (B, 276)

        return out

def build_model(X_train_1, X_train_2, X_train_3, X_train_4, Y_train,
                X_test_1, X_test_2, X_test_3, X_test_4, Y_test, Y_test_original,
                batch_size, epochs, a, time_lag):

    # Reshape 数据
    X_train_1 = X_train_1.reshape(-1, 276, time_lag - 1, 3)
    X_train_2 = X_train_2.reshape(-1, 276, time_lag - 1, 3)
    X_train_3 = X_train_3.reshape(-1, 276, time_lag - 1, 1)
    X_train_4 = X_train_4.reshape(-1, 11, time_lag - 1, 1)
    Y_train = Y_train.reshape(-1, 276)

    X_test_1 = X_test_1.reshape(-1, 276, time_lag - 1, 3)
    X_test_2 = X_test_2.reshape(-1, 276, time_lag - 1, 3)
    X_test_3 = X_test_3.reshape(-1, 276, time_lag - 1, 1)
    X_test_4 = X_test_4.reshape(-1, 11, time_lag - 1, 1)
    Y_test = Y_test.reshape(-1, 276)
    # print(X_train_1.shape, X_train_2.shape, X_train_3.shape, X_train_4.shape, Y_train.shape,
    #     X_test_1.shape, X_test_2.shape, X_test_3.shape, X_test_4.shape, Y_test.shape )

    # 转换为 Tensor
    train_dataset = TensorDataset(
        torch.tensor(X_train_1, dtype=torch.float32),
        torch.tensor(X_train_2, dtype=torch.float32),
        torch.tensor(X_train_3, dtype=torch.float32),
        torch.tensor(X_train_4, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_1, dtype=torch.float32),
        torch.tensor(X_test_2, dtype=torch.float32),
        torch.tensor(X_test_3, dtype=torch.float32),
        torch.tensor(X_test_4, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器、损失函数
    model_path = f"testresult/{50}-model.pt"

    model = MultiInputModel(time_lag)
    if epochs != 50 and os.path.exists(model_path):
        print(f"加载模型参数: {model_path}")
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(10 if epochs != 50 else epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for x1, x2, x3, x4, y in loop:
            optimizer.zero_grad()
            # print(x1.shape, x2.shape, x3.shape, x4.shape)
            output = model(x1, x2, x3, x4)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # 测试并预测
    model.eval()
    predictions = []
    with torch.no_grad():
        for x1, x2, x3, x4, _ in test_loader:
            output = model(x1, x2, x3, x4)
            predictions.append(output.numpy())

    predictions = np.vstack(predictions)

    # 反归一化
    predictions = np.round(predictions * a, 0)
    predictions[predictions < 0] = 0

    # 评估
    RMSE, R2, MAE, WMAPE = evaluate_performance(Y_test_original, predictions)

    return model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE

def save_data(path, model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch, start_time):
    print(Run_epoch)

    # 创建保存路径
    os.makedirs(path, exist_ok=True)

    # 保存指标
    np.savetxt(os.path.join(path, f"{Run_epoch}-RMSE_ALL.txt"), [RMSE])
    np.savetxt(os.path.join(path, f"{Run_epoch}-R2_ALL.txt"), [R2])
    np.savetxt(os.path.join(path, f"{Run_epoch}-MAE_ALL.txt"), [MAE])
    np.savetxt(os.path.join(path, f"{Run_epoch}-WMAPE_ALL.txt"), [WMAPE])

    # 保存模型
    torch.save(model.state_dict(), os.path.join(path, f"{Run_epoch}-model.pth"))

    # 保存预测值
    np.savetxt(os.path.join(path, f"{Run_epoch}-predictions.csv"), predictions, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(path, f"{Run_epoch}-Y_test_original.csv"), Y_test_original, delimiter=',', fmt='%.2f')

    # 保存训练耗时
    duration_time = time.time() - start_time
    np.savetxt(os.path.join(path, f"{Run_epoch}-Average_train_time.txt"), [duration_time])
    print('Total training time (s):', duration_time)

X_train_1, Y_train, X_test_1, Y_test, Y_test_original, a, b, \
X_train_2, X_test_2, X_train_3, X_test_3, X_train_4, X_test_4 = \
    Get_All_Data(TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)

Run_epoch = 50
for i in range(15):
    start_time = time.time()

    model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE = build_model(
        X_train_1, X_train_2, X_train_3, X_train_4, Y_train,
        X_test_1, X_test_2, X_test_3, X_test_4, Y_test,
        Y_test_original, batch_size=64, epochs=Run_epoch, a=a, time_lag=6
    )

    save_data("testresult/", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch, start_time)
    Run_epoch += 10