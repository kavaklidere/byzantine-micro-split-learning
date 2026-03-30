import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import copy
import csv

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

def get_dataset():
    weights = models.VGG16_Weights.DEFAULT
    preprocess = weights.transforms()

    data_dir = os.path.join('./imagenet_val_data', 'imagenet_validation')
    print(f"Loading dataset from {data_dir}...")
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
    except FileNotFoundError:
        print(f"Error: Could not find '{data_dir}'.")
    return dataset

def evaluate(device, dataloader, total_images, targeted_layer, layer_conv, std_dev, num_partition, num_iterations):
    top1_acc_total = 0.0
    top5_acc_total = 0.0

    for j in range(num_iterations):
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        
        client_index = 0
        if num_partition != 1:
            client_index = (num_partition // 2) * j
        
        if layer_conv == True:
            model.features[targeted_layer] = MicroSplitConv2d(
            model.features[targeted_layer], 
            error_std_dev=std_dev, 
            num_clients_in_layer=num_partition, 
            malicious_client_index=client_index
            )
        else: 
            model.classifier[targeted_layer] = MicroSplitLinear(
            model.classifier[targeted_layer], 
            error_std_dev=std_dev, 
            num_clients_in_layer=num_partition, 
            malicious_client_index=client_index
            )

        model = model.to(device)
        model.eval()
        
        top1_correct = 0.0
        top5_correct = 0.0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
        
                outputs = model(images)
                
                _, top5_preds = outputs.topk(5, 1, True, True)
                top5_preds = top5_preds.t()
                
                correct_matches = top5_preds.eq(labels.view(1, -1).expand_as(top5_preds))
        
                top1_correct += correct_matches[:1].reshape(-1).float().sum(0, keepdim=True).item()
                top5_correct += correct_matches[:5].reshape(-1).float().sum(0, keepdim=True).item()

                if (i + 1) % 50 == 0:
                    print(f"Processed batch {i + 1} / {len(dataloader)}...")

        top1_acc = top1_correct / total_images
        top5_acc = top5_correct / total_images

        top1_acc_total += top1_acc
        top5_acc_total += top5_acc
        
    top1_acc_total /= num_iterations
    top5_acc_total /= num_iterations

    # Return as percentages (e.g., 71.58) to match your plotting script expectations
    return top1_acc_total * 100, top5_acc_total * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full evaluation on: {device}")
    
    dataset = get_dataset()
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    total_images = len(dataset)

    print(f"Success! Found {len(dataset.classes)} classes and {total_images} total images.")

    targeted_layer = 0 # dense_1
    experiment_name = "vgg16_dense_1"
    std_devs = [0.05, 0.1, 0.25, 0.5]
    num_partitions = [1, 2, 4, 8, 16]
    num_iterations = 2

    # Initialize data structures to hold our CSV rows
    top1_data = {std: [] for std in std_devs}
    top5_data = {std: [] for std in std_devs}

    for std_dev in std_devs:
        for num_partition in num_partitions:
            print(f"\nEvaluating std_dev: {std_dev}, num_partition: {num_partition}")
            t1, t5 = evaluate(
                device=device, dataloader=dataloader, total_images=total_images, 
                targeted_layer=targeted_layer, layer_conv=False,
                std_dev=std_dev, num_partition=num_partition, num_iterations=num_iterations
            )
            
            print(f"Result -> Top-1: {t1:.2f}%, Top-5: {t5:.2f}%")
            top1_data[std_dev].append(t1)
            top5_data[std_dev].append(t5)

    # ==========================================
    # Write directly to CSV
    # ==========================================
    os.makedirs("results", exist_ok=True)
    top1_file = f"results/{experiment_name}_top_1.csv"
    top5_file = f"results/{experiment_name}_top_5.csv"
    
    header = ["std_dev \\ num_partitions"] + num_partitions

    # Write Top-1
    with open(top1_file, mode='w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(header)
        for std_dev in std_devs:
            # Rounding to 2 decimal places for a cleaner file
            row = [std_dev] + [round(val, 2) for val in top1_data[std_dev]]
            writer.writerow(row)

    # Write Top-5
    with open(top5_file, mode='w', newline='') as f5:
        writer = csv.writer(f5)
        writer.writerow(header)
        for std_dev in std_devs:
            row = [std_dev] + [round(val, 2) for val in top5_data[std_dev]]
            writer.writerow(row)

    print(f"\nAll results saved to {top1_file} and {top5_file}")

if __name__ == "__main__":
    main()