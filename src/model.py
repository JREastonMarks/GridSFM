import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class RegionalGraphormer(nn.Module):
    def __init__(self, 
                 num_node_features=5, 
                 num_edge_features=3, 
                 hidden_dim=128, 
                 num_heads=4, 
                 num_layers=4, 
                 output_dim=2):
        """
        PyTorch Geometric attention model leveraging grid topologies.
        """
        super(RegionalGraphormer, self).__init__()
        
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.edge_encoder = nn.Linear(num_edge_features, hidden_dim)
        
        # We embed the shortest path distances. Max distance is arbitrarily set to 512 for now.
        self.spatial_embedding = nn.Embedding(512, hidden_dim)
        
        # Transformer-based Graph Convolution layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    dropout=0.1
                )
            )
            
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, spatial_encoding, shortest_path=None):
        # 1. Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 2. Incorporate spatial encodings (Centrality-style embedding)
        # We sum the distances from each node to all others as a rough centrality metric,
        # embed it, and add to the initial node features. 
        # (A true dense Graphormer would add this to the attention matrix directly)
        if spatial_encoding is not None:
            # Clamping distances to max embedding size
            clamped_dist = torch.clamp(spatial_encoding, 0, 511)
            # Take the mean distance of a node to all other nodes in the graph
            mean_dist = clamped_dist.float().mean(dim=1).long()
            spatial_bias = self.spatial_embedding(mean_dist)
            x = x + spatial_bias
        
        # 3. Message Passing with Attention
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = torch.nn.functional.gelu(x)
            x = x + residual # Residual connection
            
        # 4. Predict AC/DC states for each node
        out = self.predictor(x)
        return out
