import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import anomaly_monitor as Monitor
import copy
import csv
import malicious_layer as malicious

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

def evaluate(device, dataloader, total_images, base_model, targeted_layer, layer_conv, std_dev, num_partition, num_iterations):
    top1_acc_total = 0.0
    top5_acc_total = 0.0

    for j in range(num_iterations):
        model = copy.deepcopy(base_model)
        
        client_index = 0
        if num_partition != 1:
            client_index = (num_partition // 2) * j
        
        # Inject the Malicious Layer
        if layer_conv:
            model.features[targeted_layer] = malicious.MicroSplitConv2d(
                model.features[targeted_layer], 
                error_std_dev=std_dev, 
                num_clients_in_layer=num_partition, 
                malicious_client_index=client_index
            )
        else: 
            model.classifier[targeted_layer] = malicious.MicroSplitLinear(
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

        top1_acc = top1_correct / total_images
        top5_acc = top5_correct / total_images

        top1_acc_total += top1_acc
        top5_acc_total += top5_acc

        del model
        torch.cuda.empty_cache()

    return (top1_acc_total / num_iterations) * 100, (top5_acc_total / num_iterations) * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full evaluation on: {device}")
    
    dataset = get_dataset()
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    total_images = len(dataset)

    # ==========================================
    # VGG16 CONV1-2 CONFIGURATION
    # ==========================================
    base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    targeted_layer = 2      # Index 2 is conv1-2 in model.features
    layer_conv = True       # We are targeting convolutions
    experiment_name = "vgg16_conv1_2_monitored"
    std_devs = [0.05, 0.1, 0.25, 0.5]
    num_partitions = [1, 2, 4, 8, 16]  # Specific to your VGG16 architecture tests
    num_iterations = 2

    # Initialize data structures to hold our CSV rows
    top1_data = {std: [] for std in std_devs}
    top5_data = {std: [] for std in std_devs}

    for std_dev in std_devs:
        for num_partition in num_partitions:
            print(f"\nEvaluating std_dev: {std_dev}, num_partition: {num_partition}")
            
            t1, t5 = evaluate(
                device=device, 
                dataloader=dataloader,
                base_model = base_model,
                total_images=total_images,
                targeted_layer=targeted_layer, 
                layer_conv=layer_conv,
                std_dev=std_dev, 
                num_partition=num_partition, 
                num_iterations=num_iterations,
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