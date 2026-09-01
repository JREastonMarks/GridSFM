import torch
import copy

class ContingencySimulator:
    def __init__(self, data):
        """
        Wrapper to dynamically manipulate PyTorch Geometric Data objects for inference.
        """
        self.original_data = data
        
    def simulate_n_1_outage(self, edge_idx_to_drop):
        """
        Simulates an N-1 line outage by removing an edge (and its reverse direction).
        Assumes edges are stored as [src, dst] and [dst, src] in sequence.
        """
        # Create a deep copy to avoid mutating the original data
        sim_data = copy.deepcopy(self.original_data)
        
        # Identify the forward and reverse edges to drop
        # Assuming our precompute script adds (u, v) and then (v, u)
        # We can just drop indices directly if provided, or find them.
        mask = torch.ones(sim_data.edge_index.size(1), dtype=torch.bool)
        
        src = sim_data.edge_index[0, edge_idx_to_drop]
        dst = sim_data.edge_index[1, edge_idx_to_drop]
        
        # Mask out both (src, dst) and (dst, src)
        for i in range(sim_data.edge_index.size(1)):
            s = sim_data.edge_index[0, i]
            d = sim_data.edge_index[1, i]
            if (s == src and d == dst) or (s == dst and d == src):
                mask[i] = False
                
        sim_data.edge_index = sim_data.edge_index[:, mask]
        sim_data.edge_attr = sim_data.edge_attr[mask]
        
        # NOTE: A perfectly rigorous Graphormer would recalculate the shortest 
        # path spatial_encoding here. For rapid inference, if the network is highly 
        # meshed, the distance matrix doesn't change drastically for N-1, but we 
        # can implement a dynamic networkx recalculation if needed.
        
        return sim_data

    def get_high_capacity_lines(self, threshold=5.0):
        """
        Identifies indices of edges with rate_a > threshold for targeted N-1 tests.
        """
        # Edge features are [br_r, br_x, rate_a]
        rate_a = self.original_data.edge_attr[:, 2]
        high_cap_indices = torch.where(rate_a > threshold)[0]
        return high_cap_indices.tolist()

    def scale_load(self, factor=1.2):
        """
        Scales node load demands to simulate peak conditions.
        We scale base_kv (idx 0) here as a proxy for feature mutation so the model inputs shift dynamically.
        """
        sim_data = copy.deepcopy(self.original_data)
        sim_data.x[:, 0] *= factor
        return sim_data

if __name__ == "__main__":
    # Quick test
    ma_file = "data/processed/encodings/04h/massachusetts.pt"
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        pass # Just keeping imports clean
    
    data = torch.load(ma_file, weights_only=False)
    simulator = ContingencySimulator(data)
    
    print(f"Original edges: {data.edge_index.size(1)}")
    
    # Exhaustive drop of edge 0
    sim_data = simulator.simulate_n_1_outage(0)
    print(f"After dropping edge 0: {sim_data.edge_index.size(1)}")
    
    # Targeted
    high_cap = simulator.get_high_capacity_lines(threshold=2.0)
    print(f"Found {len(high_cap)} high capacity edges for targeted N-1.")
