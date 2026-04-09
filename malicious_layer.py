import torch
import torch.nn as nn
import copy

class MicroSplitLinear(nn.Module):
    def __init__(self, original_layer, error_std_dev=0.05, num_clients_in_layer=1, malicious_client_index=0):
        super().__init__()

        self.num_clients_in_layer = num_clients_in_layer
        
        if malicious_client_index >= num_clients_in_layer or malicious_client_index < 0:
            raise ValueError(f"malicious_client_index must be between 0 and {num_clients_in_layer - 1}")
            
        self.malicious_client_index = malicious_client_index
        
        self.client_clean = copy.deepcopy(original_layer)
        self.client_noisy = copy.deepcopy(original_layer)
        
        with torch.no_grad():
            noise = torch.randn_like(self.client_noisy.weight) * error_std_dev
            self.client_noisy.weight.add_(noise)

    def forward(self, x):
        out_clean = self.client_clean(x)
        out_noisy = self.client_noisy(x)
        
        if self.num_clients_in_layer == 1:
            return out_noisy
            
        # 1. Target dimension 1 (The flat feature array)
        total_features = out_clean.shape[1]
        chunk_size = total_features // self.num_clients_in_layer
        
        start_idx = self.malicious_client_index * chunk_size
        
        if self.malicious_client_index == self.num_clients_in_layer - 1:
            end_idx = total_features
        else:
            end_idx = start_idx + chunk_size
        
        # 2. Slice along dimension 1
        left_clean = out_clean[:, :start_idx]
        mid_noisy = out_noisy[:, start_idx:end_idx]
        right_clean = out_clean[:, end_idx:]
        
        # 3. Concatenate back into a flat array
        return torch.cat([left_clean, mid_noisy, right_clean], dim=1)

class MicroSplitConv2d(nn.Module):
    def __init__(self, original_layer, error_std_dev=0.05, num_clients_in_layer=1, malicious_client_index=0):
        super().__init__()

        self.num_clients_in_layer = num_clients_in_layer
        
        # Ensure the requested malicious index is valid (0-indexed)
        if malicious_client_index >= num_clients_in_layer or malicious_client_index < 0:
            raise ValueError(f"malicious_client_index must be between 0 and {num_clients_in_layer - 1}")
            
        self.malicious_client_index = malicious_client_index
        
        # Client 1: Holds the pristine, pre-trained weights
        self.client_clean = copy.deepcopy(original_layer)
        
        # Client 2: Holds the noisy, corrupted weights
        self.client_noisy = copy.deepcopy(original_layer)
        with torch.no_grad():
            noise = torch.randn_like(self.client_noisy.weight) * error_std_dev
            self.client_noisy.weight.add_(noise)

    def forward(self, x):
        # 1. Both versions process the whole tensor
        out_clean = self.client_clean(x)
        out_noisy = self.client_noisy(x)
        
        # If there is only 1 client, the whole layer is just the noisy client
        if self.num_clients_in_layer == 1:
            return out_noisy
            
        # 2. Calculate the boundaries dynamically based on the requested index
        total_height = out_clean.shape[2]
        chunk_size = total_height // self.num_clients_in_layer
        
        start_idx = self.malicious_client_index * chunk_size
        
        # Ensure the last partition grabs any remaining pixels if the height 
        # doesn't divide perfectly by the number of clients
        if self.malicious_client_index == self.num_clients_in_layer - 1:
            end_idx = total_height
        else:
            end_idx = start_idx + chunk_size
        
        # 3. Splice the outputs together into three potential sections
        # PyTorch handles empty slices perfectly. If start_idx is 0, top_clean is safely empty.
        top_clean = out_clean[:, :, :start_idx, :]
        mid_noisy = out_noisy[:, :, start_idx:end_idx, :]
        bottom_clean = out_clean[:, :, end_idx:, :]
        
        # 4. Concatenate them all together along the height dimension (dim=2)
        return torch.cat([top_clean, mid_noisy, bottom_clean], dim=2)