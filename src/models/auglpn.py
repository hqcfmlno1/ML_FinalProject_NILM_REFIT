import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np



class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv1x3 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_mask = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
    def forward(self, F_map):
        feat1 = self.conv1x1(F_map)
        feat2 = self.conv1x3(F_map)
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        mask = torch.max(F_map, dim=2, keepdim=True)[0]  # Shape: (batch_size, in_channels, 1)
        mask = self.conv_mask(mask)                     # Shape: (batch_size, in_channels, 1)
        mask = F.hardsigmoid(mask)                      
        out = (feat1_norm * mask) * feat2_norm  # Shape: (batch_size, in_channels, seq_len)
        return out


class LFPN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # C2, C3, C4, C5 giảm chuỗi thời gian đi một nửa sau mỗi tầng
        self.conv_c2 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # 300 -> 150
        self.conv_c3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)           # 150 -> 75
        self.conv_c4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)          # 75 -> 38
        self.conv_c5 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)         # 38 -> 19
        
        # Top-down Pathway
        # Sử dụng Conv1d 1x1 để ép số kênh các nhánh ngang về 48
        self.lateral_c5 = nn.Conv1d(256, 48, kernel_size=1)
        self.lateral_c4 = nn.Conv1d(128, 48, kernel_size=1)
        self.lateral_c3 = nn.Conv1d(64, 48, kernel_size=1)
        self.lateral_c2 = nn.Conv1d(32, 48, kernel_size=1)
        
        # Lớp Conv1d 3x3 làm mịn đặc trưng sau khi Up-sampling
        self.smooth_p4 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv1d(48, 48, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 300)
        # Bottom-up pathway 
        c2 = F.relu(self.conv_c2(x))       # (batch_size, 32, 150)
        c3 = F.relu(self.conv_c3(c2))      # (batch_size, 64, 75)
        c4 = F.relu(self.conv_c4(c3))      # (batch_size, 128, 38)
        c5 = F.relu(self.conv_c5(c4))      # (batch_size, 256, 19)
        
        # Top-down pathway 
        p5 = self.lateral_c5(c5)           # (batch_size, 48, 19)
        
        p5_up = F.interpolate(p5, size=c4.shape[2], mode='linear', align_corners=False) # (batch_size, 48, 38)
        p4 = self.smooth_p4(p5_up) + self.lateral_c4(c4)                                # (batch_size, 48, 38)
        
        p4_up = F.interpolate(p4, size=c3.shape[2], mode='linear', align_corners=False) # (batch_size, 48, 75)
        p3 = self.smooth_p3(p4_up) + self.lateral_c3(c3)                                # (batch_size, 48, 75)
        
        p3_up = F.interpolate(p3, size=c2.shape[2], mode='linear', align_corners=False) # (batch_size, 48, 150)
        p2 = self.smooth_p2(p3_up) + self.lateral_c2(c2)                                # (batch_size, 48, 150)
        
        return c2, c3, c4, c5, p2, p3, p4, p5
    

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )
        # Pointwise Conv1d 1x1
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(self.act(x))


class TemporalBranchR(nn.Module):
    def __init__(self):
        super().__init__()
        self.dsc3_k3 = DepthwiseSeparableConv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dsc3_k5 = DepthwiseSeparableConv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.reduce_c3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        self.bigru_c3 = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.dsc4_k3 = DepthwiseSeparableConv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.dsc4_k5 = DepthwiseSeparableConv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.reduce_c4 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.bigru_c4 = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        
    def forward(self, c3, c4):
        out_dsc3 = self.dsc3_k3(c3) + self.dsc3_k5(c3) # (B, 64, 75)
        out_red3 = self.reduce_c3(out_dsc3)            # (B, 32, 75)
        out_perm3 = out_red3.permute(0, 2, 1)          # (B, 75, 32)
        gru3_out, _ = self.bigru_c3(out_perm3)         # (B, 75, 64)
        gru3_out = gru3_out.permute(0, 2, 1)           # (B, 64, 75)
        
        # Downsample đầu ra BiGRU của C3 từ chiều dài 75 về 38 để cộng được với nhánh C4
        gru3_out_down = F.adaptive_max_pool1d(gru3_out, output_size=c4.shape[2]) # (B, 64, 38)
        
        out_dsc4 = self.dsc4_k3(c4) + self.dsc4_k5(c4) # (B, 128, 38)
        out_red4 = self.reduce_c4(out_dsc4)            # (B, 32, 38)
        out_perm4 = out_red4.permute(0, 2, 1)          # (B, 38, 32)
        gru4_out, _ = self.bigru_c4(out_perm4)         # (B, 38, 64)
        gru4_out = gru4_out.permute(0, 2, 1)           # (B, 64, 38)
        
        feat_temporal = gru3_out_down + gru4_out       # (B, 64, 38)
        return feat_temporal
    

class SpatialBranchL(nn.Module):
    def __init__(self):
        super().__init__()
        self.am = EnhancedAttentionModule(in_channels=48)       
        self.down1_to_2 = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1) # 150 -> 75
        self.down2_to_3 = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1) # 75 -> 38
        self.lateral_p3 = nn.Conv1d(48, 48, kernel_size=1)
        self.dconv1 = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, padding=2, dilation=2)
        self.dconv2 = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, padding=4, dilation=4)
        
    def forward(self, p2, p3, p4):
        # p2: (B, 48, 150), p3: (B, 48, 75), p4: (B, 48, 38)
        f_deep1 = self.am(p2)                                    # (B, 48, 150)
        f_deep2_proj = self.down1_to_2(f_deep1)                 # (B, 48, 75)
        f_deep2 = f_deep2_proj + self.lateral_p3(p3)                              # (B, 48, 75)
        f_deep3_proj = self.down2_to_3(f_deep2)                 # (B, 48, 38/37)
        f_deep3_proj = F.adaptive_max_pool1d(f_deep3_proj, output_size=p4.shape[2]) # Khớp chính xác length p4 -> (B, 48, 38)
        f_deep3 = f_deep3_proj + p4                              # (B, 48, 38)        
        out_dconv1 = F.relu(self.dconv1(f_deep1))                # (B, 48, 150)
        out_dconv1_down = F.adaptive_max_pool1d(out_dconv1, output_size=f_deep2.shape[2]) # (B, 48, 75)
        feat_joint2 = out_dconv1_down + f_deep2                  # (B, 48, 75)
        out_dconv2 = F.relu(self.dconv2(feat_joint2))            # (B, 48, 75)
        out_dconv2_down = F.adaptive_max_pool1d(out_dconv2, output_size=f_deep3.shape[2]) # (B, 48, 38)
        feat_joint3 = out_dconv2_down + f_deep3                  # (B, 48, 38)
        return feat_joint3


class AugLPNNILM(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.lfpn = LFPN(in_channels=in_channels)
        self.spatial_L = SpatialBranchL()
        self.temporal_R = TemporalBranchR()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=112 * 38, out_features=56)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=56, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1) # Sequence-to-Point (S2P)
        
    def forward(self, x, verbose=False):
        # x: (B, 1, 300)
        if verbose:
            print(f"Input shape: {x.shape}")
        c2, c3, c4, c5, p2, p3, p4, p5 = self.lfpn(x)
        if verbose:
            print(f"--- LFPN Bottom-Up ---")
            print(f"C2 shape: {c2.shape}")
            print(f"C3 shape: {c3.shape}")
            print(f"C4 shape: {c4.shape}")
            print(f"C5 shape: {c5.shape}")
            print(f"--- LFPN Top-Down ---")
            print(f"P5 shape: {p5.shape}")
            print(f"P4 shape: {p4.shape}")
            print(f"P3 shape: {p3.shape}")
            print(f"P2 shape: {p2.shape}")

        feat_L = self.spatial_L(p2, p3, p4) # (B, 48, 38)
        if verbose:
            print(f"--- Nhánh Trái L (Spatial) ---")
            print(f"Spatial L (feat_joint3) shape: {feat_L.shape}")

        feat_R = self.temporal_R(c3, c4)   # (B, 64, 38)
        if verbose:
            print(f"--- Nhánh Phải R (Temporal) ---")
            print(f"Temporal R BiGRU shape: {feat_R.shape}")            
        feat_concat = torch.cat([feat_L, feat_R], dim=1) # (B, 112, 38)
        if verbose:
            print(f"--- Khối hòa trộn (Concatenate) ---")
            print(f"Concatenation shape: {feat_concat.shape}")

        x_flat = self.flatten(feat_concat)
        x_fc1 = self.dropout(self.relu(self.fc1(x_flat)))
        x_fc2 = self.dropout(self.relu(self.fc2(x_fc1)))
        out = self.fc3(x_fc2) # (B, 1)
        
        if verbose:
            print(f"--- Output ---")
            print(f"Final Output shape: {out.shape}")
            
        return out
    

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
        target_watt = target_scaled * (self.appliance_max - self.appliance_min) + self.appliance_min   # chuẩn hóa ngược để xem giá trị nguyên bản có lớn hơn 0 ko
        weights = torch.where(target_watt > self.threshold_watt, self.weight_factor, self.weight_factor_off)     # tạo ra 1 tensor cùng shape nếu tmdk thì thay bởi giá trị đầu tiên ko thì 1.0
        weighted_loss = loss * weights
        return weighted_loss.mean()
    

def train_auglpn(epochs, save_dir, app_num, threshold, app_max, app_min, train_loader, test_loader, train_mean, train_std, device, lr = 5e-4, taking_threshold = np.inf, weight_factor=1.0, weight_factor_off = 1.0):
    ignore_threshold = threshold[app_num - 1]
    model = AugLPNNILM(in_channels=1).to(device)
    
    if os.path.exists(save_dir):
        model.load_state_dict(torch.load(save_dir, map_location=device))
                
    criterion = WeightedMAELoss(appliance_min = app_min[app_num - 1], appliance_max = app_max[app_num - 1], weight_factor = weight_factor, threshold_watt = ignore_threshold, weight_factor_off = weight_factor_off) # Sử dụng L1 Loss theo yêu cầu
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    min_eval_loss = taking_threshold
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            # Chuẩn hóa đầu vào Z-score
            data = (data - train_mean) / train_std
            # Chuẩn hóa đầu ra Min-Max cho thiết bị cụ thể
            target = target[:, [app_num - 1]]
            target = (target - app_min[app_num - 1]) / (app_max[app_num - 1] - app_min[app_num - 1])
            data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / (batch_idx + 1)
        print(f'EPOCH: {epoch+1}, average train loss (L1): {avg_train_loss:.6f}')

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data_eval, target_eval) in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval]")):
                # Chuẩn hóa đầu vào Z-score
                data_eval = (data_eval - train_mean) / train_std
               # Chuẩn hóa đầu ra Min-Max
                target_eval = target_eval[:, [app_num - 1]]
                target_eval = (target_eval - app_min[app_num - 1]) / (app_max[app_num - 1] - app_min[app_num - 1])                
                data_eval, target_eval = data_eval.to(device).to(torch.float32), target_eval.to(device).to(torch.float32)
                pred = model(data_eval)
                eval_loss += criterion(pred, target_eval).item()               
            avg_eval_loss = eval_loss / (batch_idx + 1)            
            if avg_eval_loss < min_eval_loss:
                min_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), save_dir)
                print(f'>>> new best model with eval Loss: {avg_eval_loss:.6f} <<<')               
            print(f'EPOCH: {epoch+1}, average eval loss: {avg_eval_loss:.6f}')           
    print(f'Done training model for appliance {app_num}')


