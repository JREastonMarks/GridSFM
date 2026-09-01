import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from src.model import RegionalGraphormer
from src.dataset import GridSFMDataset

def train_model(data_dir="data/processed/encodings", epochs=500, batch_size=1):
    if not torch.backends.mps.is_available():
        print("MPS backend not found! Falling back to CPU.")
        device = torch.device("cpu")
    else:
        print("MPS backend found. Using Apple Silicon acceleration.")
        device = torch.device("mps")

    # Load dataset
    print(f"Loading full dataset from {data_dir}...")
    dataset = GridSFMDataset(root_dir=data_dir)
    print(f"Loaded {len(dataset)} graphs across the region and horizons.")
    
    # We use batch_size 2 since we have 4 graphs (MA 04h, MA 16h, NE 04h, NE 16h)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = RegionalGraphormer(
        num_node_features=5, 
        num_edge_features=3, 
        hidden_dim=128, 
        output_dim=2 
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('mps') 
    criterion = nn.MSELoss()

    # Set up checkpointing
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    model.train()
    print(f"Starting training loop for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('mps'):
                # Note: PyG batches graphs into a single giant disconnected graph
                # which means batch.spatial_encoding might need special care if used directly.
                # Since we do node centrality embedding in model.py, this works safely per node.
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.spatial_encoding)
                loss = criterion(outputs, batch.y)
            
            # Backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * batch.num_graphs
            
        epoch_loss /= len(dataset)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'stats': dataset.stats # Save normalization stats so decoder can use them later
            }, os.path.join(checkpoint_dir, 'best_model.pth'))

    print(f"\nTraining complete. Best loss: {best_loss:.6f}. Model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    train_model()
