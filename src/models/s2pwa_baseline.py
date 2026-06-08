import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np

class S2PwAModel(nn.Module):
    def __init__(self, window_length = 300, in_channels = 1, filter_nums = 32, kernel_size = 3, hidden_dim = 64):
        super().__init__()
        self.window_length = window_length

        # Encoder
        self.conv1 = nn.Conv1d(in_channels, out_channels = filter_nums, kernel_size = kernel_size, padding = kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels  = filter_nums, kernel_size = kernel_size, padding = kernel_size//2)
        self.bilstm = nn.LSTM(input_size = 32, hidden_size = hidden_dim, batch_first = True, bidirectional = True)
        bilstm_out_dim = hidden_dim * 2 # concat 2 vector của hidden state theo 2 chiều

        # Attention
        self.v_a = nn.Linear(hidden_dim, 1, bias = False)
        self.att_dense = nn.Linear(bilstm_out_dim, hidden_dim)
        self.att_softmax = nn.Softmax(dim = 1)
        
        # thay đổi kiến trúc cũ
        # đầu ra của attention là batch, bilstm_out_dim
        self.fc1 = nn.Linear(bilstm_out_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,1)
    
    def forward(self, x):
        # x sau khi lấy ra từ dataloader đang có dạng batch, channels, length
        x = self.conv1(x)
        x = self.conv2(x)
        # lstm mong đợi đầu vào dạng batch, length, channels
        x = x.permute(0,2,1) 
        h_t, _ = self.bilstm(x) 
        # h_t có dạng batch, length, hidden_dim*2
        energy = nn.Tanh()(self.att_dense(h_t)) # batch, length, hidden_dim
        score = self.v_a(energy) # batch, length, 1 (có thể hiểu là có l hidden state mối hidden state có 1 trọng số riêng để thể hiện sự đóng góp)
        alpha_t = self.att_softmax(score) # batch, length, 1
        c = (h_t * alpha_t).sum(dim = 1) # batch, hidden_dim * 2
        x = self.fc1(c) # batch, 64
        x = self.relu(x) 
        x = self.fc2(x) # batch, 1
        return x


class WeightedMAELoss(nn.Module):
    def __init__(self, weight_factor=15.0, weight_factor_off = 1.0, threshold_watt=10.0, appliance_min=0.0, appliance_max=1.0):
        super(WeightedMAELoss, self).__init__()
        self.weight_factor = weight_factor
        self.threshold_watt = threshold_watt
        self.appliance_min = appliance_min
        self.appliance_max = appliance_max
        self.weight_factor_off = weight_factor_off

    def forward(self, pred, target_scaled):
        loss = torch.abs(pred - target_scaled)
        target_watt = target_scaled * (self.appliance_max - self.appliance_min) + self.appliance_min   
        weights = torch.where(target_watt > self.threshold_watt, self.weight_factor, self.weight_factor_off)     
        weighted_loss = loss * weights
        return weighted_loss.mean()
    

def trainer(epochs, save_dir, app_num, app_min, app_max, train_mean, train_std, device, train_loader, test_loader, lr = 1e-4, weight_factor=15.0, weight_factor_off = 1.0, threshold_watt = 15.0):
    model = S2PwAModel().to(device)
    if os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir, weights_only=True))
    criterion = WeightedMAELoss(appliance_min = app_min[app_num - 1], appliance_max = app_max[app_num - 1], weight_factor = weight_factor, threshold_watt = threshold_watt, weight_factor_off = weight_factor_off)                 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    min_eval_loss = np.inf
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = (data - train_mean)/train_std
            target = target[:,[app_num - 1]]
            target = (target - app_min[app_num - 1])/(app_max[app_num - 1] - app_min[app_num - 1])
            data, target = data.to(device).to(torch.float), target.to(device).to(torch.float)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/(batch_idx + 1)
        print(f'EPOCH: {epoch+1}, average train loss: {avg_train_loss}')

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data_eval, target_eval) in enumerate(tqdm(test_loader)):
                data_eval = (data_eval - train_mean)/train_std
                data_eval = data_eval.to(device)
                target_eval = target_eval[:,[app_num - 1]].to(device)
                target_eval = (target_eval - app_min[app_num - 1])/(app_max[app_num - 1] - app_min[app_num - 1])
                pred = model(data_eval)
                eval_loss += criterion(pred, target_eval).item()
            avg_eval_loss = eval_loss/(batch_idx + 1)
            if avg_eval_loss < min_eval_loss:
                min_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), save_dir)
            print(f'EPOCH: {epoch+1}, average eval loss: {avg_eval_loss}')
    print(f'Done training model for appliance {app_num}')