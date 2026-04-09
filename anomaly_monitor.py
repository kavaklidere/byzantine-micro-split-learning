import torch

class AnomalyMonitor:
    def __init__(self, threshold_sigma=4.0):
        self.threshold_sigma = threshold_sigma
        
        # Raw memory for profiling
        self.raw_logs = []
        
        # Refined signatures
        self.mean = None
        self.std = None

    def calibrate_input(self, tensor):
        """Pass the raw incoming tensor during the calibration phase."""
        # More Pythonic/Torch-idiomatic way to flatten all dims except batch
        flattened = tensor.flatten(start_dim=1)
        
        # Calculate variance across features, keeping it as a PyTorch tensor
        variances = torch.var(flattened, dim=1)
        
        # Append the 1D tensor of batch variances directly
        self.raw_logs.append(variances.detach())

    def finalize_profile(self):
        """Run this once after calibration is finished."""
        if not self.raw_logs:
            raise ValueError("No calibration data collected!")

        # Concatenate all batch tensors into one massive 1D tensor
        all_variances = torch.cat(self.raw_logs)

        # Pure PyTorch math, extracting the final scalar with .item()
        self.mean = torch.mean(all_variances).item()
        self.std = torch.std(all_variances).item()
        
        if self.std == 0:
            self.std = 1e-8 
            
        self.raw_logs.clear()  # Free memory
        print(f"[Security] Profile finalized -> Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def is_anomalous(self, tensor):
        """Pass incoming tensors here during the test phase. Returns True if attacked."""
        if self.mean is None or self.std is None:
            raise ValueError("The monitor was not calibrated!")

        flattened = tensor.flatten(start_dim=1)
        current_vars = torch.var(flattened, dim=1)

        # Broadcasted tensor subtraction and division
        z_scores = (current_vars - self.mean) / self.std
        
        # torch.any() returns a boolean tensor, .item() converts it to a standard Python bool
        return torch.any(z_scores > self.threshold_sigma).item()


def run_calibration(device, model, dataloader, targeted_layer_index, is_conv, monitor):
    model = model.to(device)
    model.eval()

    # The activation function (ReLU) immediately follows the target layer
    post_activation_index = targeted_layer_index + 1

    block_name = "features" if is_conv else "classifier"
    print(f"\n--- Starting Calibration Phase for model.{block_name}[{post_activation_index}] ---")

    def profiling_hook(module, input, output):
        monitor.calibrate_input(output)

    # Dynamically attach the hook to either model.features or model.classifier
    if is_conv:
        hook_handle = model.features[post_activation_index].register_forward_hook(profiling_hook)
    else:
        hook_handle = model.classifier[post_activation_index].register_forward_hook(profiling_hook)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)
            
            if (i + 1) % 10 == 0:
                print(f"Calibrated batch {i + 1} / {len(dataloader)}")

    hook_handle.remove()
    monitor.finalize_profile()
    
    print("--- Calibration Complete ---")
    return monitor