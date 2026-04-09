import torch
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

def evaluate(device, dataloader, total_images, targeted_layer, layer_conv, std_dev, num_partition, num_iterations, monitor):
    total_anomalies = 0

    for j in range(num_iterations):
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        
        client_index = 0
        if num_partition != 1:
            client_index = (num_partition // 2) * j
        
        # ==========================================
        # FALSE POSITIVE CHECK: Only inject if std_dev > 0
        # ==========================================
        if std_dev > 0.0:
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
        
        # ==========================================
        # SECURITY HOOK (Per-Image Tracking)
        # ==========================================
        def security_hook(module, input, output):
            nonlocal total_anomalies
            
            flattened = output.flatten(start_dim=1)
            current_vars = torch.var(flattened, dim=1)
            z_scores = (current_vars - monitor.mean) / monitor.std
            
            anomalies_in_batch = torch.sum(z_scores > monitor.threshold_sigma).item()
            total_anomalies += anomalies_in_batch

        post_activation_index = targeted_layer + 1
        if layer_conv:
            hook_handle = model.features[post_activation_index].register_forward_hook(security_hook)
        else:
            hook_handle = model.classifier[post_activation_index].register_forward_hook(security_hook)

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                images = images.to(device)
                _ = model(images) 

        hook_handle.remove()

    total_evaluated_images = total_images * num_iterations
    anomaly_percentage = (total_anomalies / total_evaluated_images) * 100
    
    # Dynamically change the print statement based on what we are testing
    if std_dev == 0.0:
        print(f"   -> BASELINE FALSE POSITIVES: {total_anomalies} / {total_evaluated_images} clean images incorrectly flagged.")
    else:
        print(f"   -> ATTACK DETECTED: {total_anomalies} / {total_evaluated_images} images successfully flagged.")

    return anomaly_percentage


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full evaluation on: {device}")
    
    dataset = get_dataset()
    batch_size = 64
    total_images = len(dataset)
    
    # ==========================================
    # 1. SPLIT THE DATASET (10% Calib, 90% Test)
    # ==========================================
    calib_count = int(0.10 * total_images)
    test_count = total_images - calib_count

    generator = torch.Generator().manual_seed(42)
    calib_dataset, test_dataset = random_split(
        dataset, [calib_count, test_count], generator=generator
    )

    calib_dataloader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Success! Total images: {total_images}. Split -> Calibration: {calib_count}, Testing: {test_count}")

    # ==========================================
    # VGG16 CONV1-2 CONFIGURATION
    # ==========================================
    targeted_layer = 2      
    layer_conv = True       
    experiment_name = "vgg16_conv1_2_monitored"
    
    # Separate the baseline from the attacks
    attack_std_devs = [0.05, 0.1, 0.25, 0.5]  
    num_partitions = [1, 2, 4, 8, 16]  
    num_iterations = 2

    # ==========================================
    # 2. RUN CALIBRATION PHASE
    # ==========================================
    monitor = Monitor.AnomalyMonitor(threshold_sigma=4.0)
    clean_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    monitor = Monitor.run_calibration(
        device=device, 
        model=clean_model, 
        dataloader=calib_dataloader, 
        targeted_layer_index=targeted_layer, 
        is_conv=layer_conv,
        monitor=monitor
    )

    # Initialize data structure for anomaly rates
    anomaly_data = {}

    # ==========================================
    # 3. RUN FALSE POSITIVE BASELINE (ONCE)
    # ==========================================
    print("\n--- Evaluating False Positive Baseline (Clean Data) ---")
    baseline_fpr = evaluate(
        device=device, 
        dataloader=test_dataloader,   
        total_images=test_count,      
        targeted_layer=targeted_layer, 
        layer_conv=layer_conv,
        std_dev=0.0,                  # Zero noise
        num_partition=1,              # Arbitrary, since no split occurs
        num_iterations=num_iterations,
        monitor=monitor               
    )
    print(f"Result -> False Positive Rate: {baseline_fpr:.2f}%")
    
    # Broadcast the baseline result across all partition columns
    anomaly_data[0.0] = [baseline_fpr] * len(num_partitions)

    # ==========================================
    # 4. RUN ATTACK EVALUATIONS
    # ==========================================
    for std_dev in attack_std_devs:
        anomaly_data[std_dev] = []
        for num_partition in num_partitions:
            print(f"\nEvaluating Attack std_dev: {std_dev}, num_partition: {num_partition}")
            
            anomaly_pct = evaluate(
                device=device, 
                dataloader=test_dataloader,   
                total_images=test_count,      
                targeted_layer=targeted_layer, 
                layer_conv=layer_conv,
                std_dev=std_dev, 
                num_partition=num_partition, 
                num_iterations=num_iterations,
                monitor=monitor               
            )
            
            print(f"Result -> True Positive Rate (Detection): {anomaly_pct:.2f}%")
            anomaly_data[std_dev].append(anomaly_pct)

    # ==========================================
    # Write directly to CSV
    # ==========================================
    os.makedirs("results", exist_ok=True)
    anomaly_file = f"results/{experiment_name}_anomaly_rates.csv"
    
    header = ["std_dev \\ num_partitions"] + num_partitions

    # Combine 0.0 with the rest of the std_devs for writing
    all_std_devs = [0.0] + attack_std_devs

    with open(anomaly_file, mode='w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(header)
        for std_dev in all_std_devs:
            row = [std_dev] + [round(val, 2) for val in anomaly_data[std_dev]]
            writer.writerow(row)

    print(f"\nAll results saved to {anomaly_file}")

if __name__ == "__main__":
    main()