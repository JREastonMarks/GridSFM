import time
import torch
import networkx as nx
from src.model import RegionalGraphormer
from src.decoder import GridDecoder
from src.analytics import AdvancedGridAnalytics

chkpt = torch.load('models/ma_regional_best.pt', map_location='cpu')
model = RegionalGraphormer(
    node_in_dim=5,
    edge_in_dim=3,
    hidden_dim=chkpt['model_kwargs']['hidden_dim'],
    num_layers=chkpt['model_kwargs']['num_layers'],
    num_heads=chkpt['model_kwargs']['num_heads']
)
model.load_state_dict(chkpt['model_state'])
decoder = GridDecoder()
analytics_engine = AdvancedGridAnalytics(model, decoder, chkpt['stats'], 'cpu')

from scripts.app import base_data
print(f"Nodes: {base_data.num_nodes}, Edges: {base_data.edge_index.size(1)}")

t0 = time.time()
v_viols, t_viols, _, _ = analytics_engine.run_inference(base_data)
print(f"One inference: {time.time() - t0:.4f}s")
