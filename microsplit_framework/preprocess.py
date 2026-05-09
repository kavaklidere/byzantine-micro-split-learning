from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans


def preprocess_clustering(
    model: nn.Module,
    dataloader,
    n_batches: int = 150,
    n_clusters: int = 100,
    device: torch.device | None = None,
) -> None:
    """
    Build cluster vocabularies for all clients using incremental MiniBatchKMeans.

    After each forward pass, each client's single-batch eavesdrop buffer is fed
    to partial_fit() and immediately freed — memory stays at O(1 batch) rather
    than O(n_batches). This prevents OOM on clusters when activation maps are large.

    All clients preprocess regardless of attack type, so any client can be switched
    to ClusterJumpAttack in a later experiment without re-running this step.
    """
    if device is None:
        device = next(model.parameters()).device

    kmeans_map = {
        cm.client_id: MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, n_init="auto", random_state=42)
        for cm in model.client_modules
    }

    # Accumulate samples per client until we have >= n_clusters for initialization.
    # partial_fit raises ValueError if n_samples < n_clusters on the first call.
    init_buffers: dict[str, list[np.ndarray]] = {
        cm.client_id: [] for cm in model.client_modules
    }
    initialized: set[str] = set()

    n_clients = len(model.client_modules)
    print(
        f"[Preprocess] Running {n_batches} batches — incremental clustering "
        f"for {n_clients} client(s) ({n_clusters} clusters each)..."
    )

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= n_batches:
                break

            model(images.to(device))

            for cm in model.client_modules:
                if not cm.eavesdrop_memory:
                    continue

                batch = cm.eavesdrop_memory[0]
                flat  = batch.view(batch.size(0), -1).numpy()
                cm.eavesdrop_memory.clear()

                km = kmeans_map[cm.client_id]

                if cm.client_id not in initialized:
                    init_buffers[cm.client_id].append(flat)
                    combined = np.concatenate(init_buffers[cm.client_id])
                    if len(combined) >= n_clusters:
                        km.partial_fit(combined)
                        init_buffers[cm.client_id].clear()
                        initialized.add(cm.client_id)
                else:
                    km.partial_fit(flat)

            if (i + 1) % 20 == 0:
                print(f"  Preprocessing batch {i + 1} / {n_batches}")

    # Store final centroids on each ClientModule
    for cm in model.client_modules:
        km    = kmeans_map[cm.client_id]
        shape = cm._eavesdrop_shape
        if km.cluster_centers_ is not None and shape is not None:
            print(f"  [{cm.client_id}] Storing {n_clusters} centroids...")
            cm.cluster_centroids = torch.tensor(
                km.cluster_centers_, dtype=torch.float32, device=device
            ).view(-1, *shape)
            cm.is_cluster_ready = True
        else:
            print(f"  [{cm.client_id}] No eavesdrop data — skipping.")

    print("[Preprocess] Done. All clients have cluster vocabularies ready.")
