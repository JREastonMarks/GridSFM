import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np

class AdvancedGridAnalytics:
    def __init__(self, model, decoder, stats, device):
        self.model = model
        self.decoder = decoder
        self.stats = stats
        self.device = device
        self.model.eval()

    def compute_spatial_encoding(self, sim_data):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(sim_data.num_nodes))
        G.add_edges_from(sim_data.edge_index.t().tolist())
        spd_matrix = torch.zeros((sim_data.num_nodes, sim_data.num_nodes), dtype=torch.long)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(sim_data.num_nodes):
            for j in range(sim_data.num_nodes):
                spd_matrix[i, j] = lengths[i][j] if j in lengths.get(i, {}) else 510
        return spd_matrix

    def run_inference(self, sim_data, recompute_spatial=True):
        if recompute_spatial or not hasattr(sim_data, 'spatial_encoding'):
            sim_data.spatial_encoding = self.compute_spatial_encoding(sim_data)
        x_norm = (sim_data.x - self.stats['x_mean'].to(self.device)) / (self.stats['x_std'].to(self.device) + 1e-8)
        edge_norm = (sim_data.edge_attr - self.stats['edge_mean'].to(self.device)) / (self.stats['edge_std'].to(self.device) + 1e-8)
        
        with torch.no_grad():
            outputs = self.model(x_norm, sim_data.edge_index, edge_norm, sim_data.spatial_encoding.to(self.device))
            
        va, vm = self.decoder.decode_voltage(outputs)
        vmin, vmax = sim_data.x[:, 2], sim_data.x[:, 1]
        
        v_viols = self.decoder.check_voltage_violations(vm, vmin, vmax)
        t_viols = self.decoder.check_thermal_violations(va, vm, sim_data.edge_index, sim_data.edge_attr)
        return v_viols, t_viols, va, vm

    def count_violations(self, sim_data, v_viols, t_viols):
        """Helper to accurately count violations, handling undirected double-edges and offline nodes."""
        # 1. Thermal violations: edge_index has both directions, so divide by 2
        t_count = t_viols.sum().item() // 2
        
        # 2. Voltage violations: Ignore offline/isolated nodes with 0 load
        # Count node degree to ensure it is actually connected to the grid
        degree = torch.zeros(sim_data.num_nodes, dtype=torch.long, device=self.device)
        src, dst = sim_data.edge_index[0], sim_data.edge_index[1]
        if len(src) > 0:
            degree.scatter_add_(0, src, torch.ones_like(src))
            degree.scatter_add_(0, dst, torch.ones_like(dst))
            
        # Node must be connected AND have non-zero load (not intentionally taken offline)
        active_nodes = (degree > 0) & (sim_data.x[:, 0] > 0)
        v_count = (v_viols & active_nodes).sum().item()
        
        return v_count, t_count

    def run_inference_grad(self, sim_data):
        """A fully differentiable forward pass for SCED optimization."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(sim_data.num_nodes))
        G.add_edges_from(sim_data.edge_index.t().tolist())
        spd_matrix = torch.zeros((sim_data.num_nodes, sim_data.num_nodes), dtype=torch.long)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(sim_data.num_nodes):
            for j in range(sim_data.num_nodes):
                spd_matrix[i, j] = lengths[i][j] if j in lengths.get(i, {}) else 510
        sim_data.spatial_encoding = spd_matrix
        
        x_norm = (sim_data.x - self.stats['x_mean'].to(self.device)) / (self.stats['x_std'].to(self.device) + 1e-8)
        edge_norm = (sim_data.edge_attr - self.stats['edge_mean'].to(self.device)) / (self.stats['edge_std'].to(self.device) + 1e-8)
        
        outputs = self.model(x_norm, sim_data.edge_index, edge_norm, sim_data.spatial_encoding.to(self.device))
        va, vm = self.decoder.decode_voltage(outputs)
        vmin, vmax = sim_data.x[:, 2], sim_data.x[:, 1]
        
        v_viols_soft = F.relu(vm - vmax) + F.relu(vmin - vm)
        
        # Soft differentiable thermal flow
        src, dst = sim_data.edge_index[0], sim_data.edge_index[1]
        vm_src, vm_dst = vm[src], vm[dst]
        va_src, va_dst = va[src], va[dst]
        X = sim_data.edge_attr[:, 1].to(self.device)
        rate = sim_data.edge_attr[:, 2].to(self.device)
        
        flow = (vm_src * vm_dst * torch.sin(va_src - va_dst)) / (X + 1e-6)
        t_viols_soft = F.relu(torch.abs(flow) - rate)
        
        return v_viols_soft, t_viols_soft

    def hosting_capacity(self, base_data, node_idx: int, max_mw: float = 1000.0):
        """1. Optimal Renewable Energy Siting: Finds max MW injection before first violation using Binary Search."""
        sim_data = base_data.clone()
        original_load = sim_data.x[node_idx, 0].item()
        
        low, high = 0.0, max_mw
        best_safe = 0.0
        
        for _ in range(15):
            mid = (low + high) / 2.0
            # Injecting generation is equivalent to reducing the net active load
            sim_data.x[node_idx, 0] = original_load - mid
            v_viols, t_viols, _, _ = self.run_inference(sim_data)
            v_c, t_c = self.count_violations(sim_data, v_viols, t_viols)
            
            if v_c > 0 or t_c > 0:
                high = mid  # Hit a violation, try lower injection
            else:
                low = mid   # Safe, try higher injection
                best_safe = mid
                
        return best_safe

    def nk_contingency_mc(self, base_data, num_samples=100, k=3):
        """2. Extreme Weather Stress-Testing: Monte Carlo random drops of k edges."""
        fatal_count = 0
        worst_combo = None
        max_viols = -1
        
        for _ in range(num_samples):
            sim_data = base_data.clone()
            num_edges = sim_data.edge_index.size(1)
            choices = np.random.choice(num_edges, k, replace=False)
            
            mask = torch.ones(num_edges, dtype=torch.bool)
            combo_recorded = []
            for e in choices:
                src, dst = sim_data.edge_index[0, e], sim_data.edge_index[1, e]
                combo_recorded.append((src.item(), dst.item()))
                mask[e] = False
                rev_mask = (sim_data.edge_index[0] == dst) & (sim_data.edge_index[1] == src)
                mask[rev_mask] = False
                
            sim_data.edge_index = sim_data.edge_index[:, mask]
            sim_data.edge_attr = sim_data.edge_attr[mask]
            
            v_viols, t_viols, _, _ = self.run_inference(sim_data)
            v_c, t_c = self.count_violations(sim_data, v_viols, t_viols)
            tot = v_c + t_c
            
            if tot > 0:
                fatal_count += 1
            if tot > max_viols:
                max_viols = tot
                worst_combo = combo_recorded
                
        return (fatal_count / num_samples) * 100.0, worst_combo, max_viols

    def expansion_planning(self, base_data, src: int, dst: int, rate: float = 150.0, X: float = 0.05, R: float = 0.01):
        """3. What-If Expansion Planning: Adds a new edge and calculates congestion delta."""
        v_viols_base, t_viols_base, _, _ = self.run_inference(base_data)
        vb_c, tb_c = self.count_violations(base_data, v_viols_base, t_viols_base)
        base_viols = vb_c + tb_c
        
        sim_data = base_data.clone()
        new_edge = torch.tensor([[src, dst], [dst, src]], dtype=torch.long).to(self.device)
        new_attr = torch.tensor([[R, X, rate], [R, X, rate]], dtype=torch.float32).to(self.device)
        
        sim_data.edge_index = torch.cat([sim_data.edge_index, new_edge], dim=1)
        sim_data.edge_attr = torch.cat([sim_data.edge_attr, new_attr], dim=0)
        
        v_viols_new, t_viols_new, _, _ = self.run_inference(sim_data)
        vn_c, tn_c = self.count_violations(sim_data, v_viols_new, t_viols_new)
        new_viols = vn_c + tn_c
        
        return base_viols, new_viols

    def run_sced(self, base_data, lr=5.0, steps=100):
        """4. Real-Time SCED: Gradient descent to adjust dispatch to zero-out violations."""
        sim_data = base_data.clone()
        # Only loads/generators can be adjusted, we parameterize x
        x_param = sim_data.x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_param], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            temp_data = sim_data.clone()
            temp_data.x = x_param
            
            v_soft, t_soft = self.run_inference_grad(temp_data)
            loss = v_soft.sum() + t_soft.sum()
            
            if loss.item() < 1e-4:
                break
                
            loss.backward()
            optimizer.step()
            
            # Constrain to meaningful physical boundaries
            with torch.no_grad():
                x_param[:, 0].clamp_(min=0.0) # Cannot go below 0 demand if it's purely a load bus
                
        return x_param.detach()

    def red_team_heuristic(self, base_data, max_removals: int = 2):
        """5. Adversarial Red Teaming: Greedy heuristic to find worst-case N-k cascading trigger."""
        best_removals = []
        sim_data = base_data.clone()
        
        for step in range(max_removals):
            max_viols = -1
            best_target = None
            
            # Heuristic Optimization: Only test the Top 15 edges with the highest capacity ratings
            # Testing every edge is O(E) with shortest-path recomputation causing massive slowdowns.
            edge_ratings = sim_data.edge_attr[:, 2]
            _, top_edges = torch.topk(edge_ratings, min(15, len(edge_ratings)))
            
            for e_idx in top_edges.tolist():
                temp_data = sim_data.clone()
                mask = torch.ones(temp_data.edge_index.size(1), dtype=torch.bool)
                mask[e_idx] = False
                src, dst = temp_data.edge_index[0, e_idx], temp_data.edge_index[1, e_idx]
                rev_mask = (temp_data.edge_index[0] == dst) & (temp_data.edge_index[1] == src)
                mask[rev_mask] = False
                
                temp_data.edge_index = temp_data.edge_index[:, mask]
                temp_data.edge_attr = temp_data.edge_attr[mask]
                
                v_viols, t_viols, _, _ = self.run_inference(temp_data)
                total_viols = v_viols.sum().item() + t_viols.sum().item()
                
                if total_viols > max_viols:
                    max_viols = total_viols
                    best_target = ('edge', (src.item(), dst.item()))
            
            # Heuristic Optimization: Only test the Top 15 nodes with the highest load
            node_loads = sim_data.x[:, 0]
            _, top_nodes = torch.topk(node_loads, min(15, len(node_loads)))
            
            for n_idx in top_nodes.tolist():
                if sim_data.x[n_idx, 0] == 0: continue
                temp_data = sim_data.clone()
                temp_data.x[n_idx, 0] = 0.0
                
                mask = (temp_data.edge_index[0] != n_idx) & (temp_data.edge_index[1] != n_idx)
                temp_data.edge_index = temp_data.edge_index[:, mask]
                temp_data.edge_attr = temp_data.edge_attr[mask]
                
                v_viols, t_viols, _, _ = self.run_inference(temp_data)
                total_viols = v_viols.sum().item() + t_viols.sum().item()
                
                if total_viols > max_viols:
                    max_viols = total_viols
                    best_target = ('node', n_idx)
                    
            if best_target:
                best_removals.append(best_target)
                if best_target[0] == 'edge':
                    src, dst = best_target[1]
                    mask = ~((sim_data.edge_index[0] == src) & (sim_data.edge_index[1] == dst))
                    mask = mask & ~((sim_data.edge_index[0] == dst) & (sim_data.edge_index[1] == src))
                    sim_data.edge_index = sim_data.edge_index[:, mask]
                    sim_data.edge_attr = sim_data.edge_attr[mask]
                else:
                    n_idx = best_target[1]
                    sim_data.x[n_idx, 0] = 0.0
                    mask = (sim_data.edge_index[0] != n_idx) & (sim_data.edge_index[1] != n_idx)
                    sim_data.edge_index = sim_data.edge_index[:, mask]
                    sim_data.edge_attr = sim_data.edge_attr[mask]
            else:
                break
                
        return best_removals, max_viols
