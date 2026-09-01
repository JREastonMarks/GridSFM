import os
import glob
import torch
from torch_geometric.data import Dataset

class GridSFMDataset(Dataset):
    def __init__(self, root_dir="data/processed/encodings", transform=None, pre_transform=None):
        super(GridSFMDataset, self).__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        
        # Discover all .pt files
        self.files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.files.append(os.path.join(root, file))
                    
        # Sort for deterministic ordering
        self.files.sort()
        
        self.stats = self._compute_stats()

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        
        # Apply Z-score normalization
        data.x = (data.x - self.stats['x_mean']) / (self.stats['x_std'] + 1e-8)
        data.edge_attr = (data.edge_attr - self.stats['edge_mean']) / (self.stats['edge_std'] + 1e-8)
        data.y = (data.y - self.stats['y_mean']) / (self.stats['y_std'] + 1e-8)
        
        return data
        
    def _compute_stats(self):
        """
        Computes mean and std across the entire dataset to normalize features.
        In a large dataset, you'd cache this to a JSON file.
        """
        x_all, edge_all, y_all = [], [], []
        
        for file in self.files:
            data = torch.load(file, weights_only=False)
            x_all.append(data.x)
            edge_all.append(data.edge_attr)
            y_all.append(data.y)
            
        x_all = torch.cat(x_all, dim=0)
        edge_all = torch.cat(edge_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        
        return {
            'x_mean': x_all.mean(dim=0),
            'x_std': x_all.std(dim=0),
            'edge_mean': edge_all.mean(dim=0),
            'edge_std': edge_all.std(dim=0),
            'y_mean': y_all.mean(dim=0),
            'y_std': y_all.std(dim=0)
        }
