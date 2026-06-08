import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

class S2PwaRegressor(nn.Module):
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
    

class S2qwaClassifier(nn.Module):
    def __init__(self, window_length=300):
        super().__init__()
        self.activation = nn.ReLU()     
        
        self.cv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10)
        self.bn1 = nn.BatchNorm1d(30)
        
        self.cv2 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(30)
        
        self.cv3 = nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6)
        self.bn3 = nn.BatchNorm1d(40)
        
        self.cv4 = nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(50)
        
        self.cv5 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5)
        self.bn5 = nn.BatchNorm1d(50)
        
        self.cv6 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5)
        self.bn6 = nn.BatchNorm1d(50)
        
        # tổng lượng hao hụt: 9 + 7 + 5 + 4 + 4 + 4 = 33
        l_out = window_length - 33 
        self.flatten_size = 50 * l_out   
        
        self.dense1 = nn.Linear(in_features=self.flatten_size, out_features=128)
        self.dropout1 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear(in_features=128, out_features=32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.activation(self.bn1(self.cv1(x)))
        x = self.activation(self.bn2(self.cv2(x)))
        x = self.activation(self.bn3(self.cv3(x)))
        x = self.activation(self.bn4(self.cv4(x)))
        x = self.activation(self.bn5(self.cv5(x)))
        x = self.activation(self.bn6(self.cv6(x))) 
        x = x.view(x.size(0), -1)         
        x = self.activation(self.dropout1(self.dense1(x)))
        x = self.activation(self.dropout2(self.dense2(x)))
        x = self.dense3(x) 
        return x


class MaskedL1Loss(nn.Module):
    def __init__(self, ignore_threshold, appliance_min, appliance_max):
        super(MaskedL1Loss, self).__init__()
        self.ignore_threshold = ignore_threshold
        self.appliance_min = appliance_min
        self.appliance_max = appliance_max
    
    def forward(self, pred, target_scaled):
        loss = torch.abs(pred - target_scaled)
        target_watt = target_scaled * (self.appliance_max - self.appliance_min) + self.appliance_min   
        mask = (target_watt >= self.ignore_threshold).float()
        total_non_masked = mask.sum()
        masked_loss = ((loss * mask).sum())/(total_non_masked + 1e-8)    # tránh mẫu = 0
        return masked_loss
    

def train_classifier(epochs, threshold, train_loader, test_loader, train_mean, train_std, save_dir, app_num, device, lr = 5e-4, pos_weight = torch.tensor(1.0)):
    ignore_threshold = threshold[app_num - 1]
    classifier = S2qwaClassifier().to(device)
    if os.path.exists(save_dir):
        classifier.load_state_dict(torch.load(save_dir, map_location=device))
    optimizer = torch.optim.Adam(classifier.parameters(), lr = lr, weight_decay = 1e-4) # L2 penalty
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    
    max_f1 = 0.0 
    
    for epoch in range(epochs):
        train_loss = 0.0
        classifier.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = (data - train_mean)/train_std
            target = target[:,[app_num - 1]]
            target = torch.where(target >= ignore_threshold, 1.0, 0.0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/(batch_idx + 1)
        print(f'EPOCH: {epoch+1}, average train loss of classifier: {avg_train_loss:.4f}')

        classifier.eval()
        eval_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (data_eval, target_eval) in enumerate(tqdm(test_loader)):
                data_eval = (data_eval - train_mean)/train_std
                target_eval = target_eval[:,[app_num - 1]]
                target_eval = torch.where(target_eval >= ignore_threshold, 1.0, 0.0)
                data_eval, target_eval = data_eval.to(device), target_eval.to(device)
                output = classifier(data_eval)
                loss = criterion(output, target_eval)
                eval_loss += loss.item()               
                preds = (output > 0).int().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(target_eval.int().cpu().numpy().flatten())
                
            avg_eval_loss = eval_loss/(batch_idx + 1)
            current_f1 = f1_score(all_targets, all_preds)
            
            if current_f1 > max_f1:
                max_f1 = current_f1
                torch.save(classifier.state_dict(), save_dir)
                print(f'>>> new best model at {epoch+1} with F1: {current_f1:.4f} <<<')
                
            print(f'EPOCH: {epoch+1}, average eval loss: {avg_eval_loss:.4f}, eval F1-score: {current_f1:.4f}')
            
    print(f'Done training model for appliance {app_num}')



def train_regressor(epochs, threshold, train_loader, test_loader, train_mean, train_std, app_min, app_max, save_dir, app_num, device , lr = 5e-4):
    ignore_threshold = threshold[app_num - 1]
    regressor = S2PwaRegressor().to(device)
    if os.path.exists(save_dir):
        regressor.load_state_dict(torch.load(save_dir, weights_only=True))
    criterion = MaskedL1Loss(appliance_min = app_min[app_num - 1], appliance_max = app_max[app_num - 1], ignore_threshold = ignore_threshold)                  
    optimizer = torch.optim.Adam(regressor.parameters(), lr = lr)
    min_eval_loss = np.inf
    for epoch in range(epochs):
        regressor.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = (data - train_mean)/train_std
            target = target[:,[app_num - 1]]
            target = (target - app_min[app_num - 1])/(app_max[app_num - 1] - app_min[app_num - 1])
            data, target = data.to(device).to(torch.float), target.to(device).to(torch.float)
            optimizer.zero_grad()
            output = regressor(data)
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/(batch_idx + 1)
        print(f'EPOCH: {epoch+1}, average train loss: {avg_train_loss}')

        regressor.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data_eval, target_eval) in enumerate(tqdm(test_loader)):
                data_eval = (data_eval - train_mean)/train_std
                data_eval = data_eval.to(device)
                target_eval = target_eval[:,[app_num - 1]].to(device)
                target_eval = (target_eval - app_min[app_num - 1])/(app_max[app_num - 1] - app_min[app_num - 1])
                pred = regressor(data_eval)
                eval_loss += criterion(pred, target_eval).item()
            avg_eval_loss = eval_loss/(batch_idx + 1)
            if avg_eval_loss < min_eval_loss:
                min_eval_loss = avg_eval_loss
                torch.save(regressor.state_dict(), save_dir)
            print(f'EPOCH: {epoch+1}, average eval loss: {avg_eval_loss}')
    print(f'Done training model for appliance {app_num}')