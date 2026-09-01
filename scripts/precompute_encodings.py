import os
import json
import torch
from torch_geometric.data import Data
import networkx as nx

def build_graph(model_path, result_path=None):
    with open(model_path, 'r') as f:
        model_data = json.load(f)
        
    res_data = None
    if result_path and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            res_data = json.load(f)

    buses = model_data.get('bus', {})
    branches = model_data.get('branch', {})
    
    # Map original bus_i to contiguous indices 0..N-1
    bus_ids = sorted(list(buses.keys()), key=lambda x: int(x))
    bus_map = {int(k): i for i, k in enumerate(bus_ids)}
    
    num_nodes = len(bus_ids)
    
    # Node features: [base_kv, vmax, vmin, lat, lon]
    x = torch.zeros((num_nodes, 5), dtype=torch.float32)
    y = torch.zeros((num_nodes, 2), dtype=torch.float32) # [va, vm] from results
    
    for k, v in buses.items():
        idx = bus_map[int(k)]
        x[idx, 0] = v.get('base_kv', 0.0)
        x[idx, 1] = v.get('vmax', 1.0)
        x[idx, 2] = v.get('vmin', 0.9)
        x[idx, 3] = v.get('lat', 0.0)
        x[idx, 4] = v.get('lon', 0.0)
        
        if res_data:
            sol_bus = res_data.get('solution', {}).get('bus', {}).get(k, {})
            y[idx, 0] = sol_bus.get('va', 0.0)
            y[idx, 1] = sol_bus.get('vm', 1.0)
            
    # Edges
    edge_index = []
    edge_attr = []
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for k, v in branches.items():
        f_bus = v.get('f_bus')
        t_bus = v.get('t_bus')
        if f_bus not in bus_map or t_bus not in bus_map:
            continue
        
        f_idx = bus_map[f_bus]
        t_idx = bus_map[t_bus]
        
        # Add both directions for undirected graph
        edge_index.append([f_idx, t_idx])
        edge_index.append([t_idx, f_idx])
        
        # Edge features: [br_r, br_x, rate_a]
        attr = [v.get('br_r', 0.0), v.get('br_x', 0.0), v.get('rate_a', 0.0)]
        edge_attr.append(attr)
        edge_attr.append(attr)
        
        G.add_edge(f_idx, t_idx)
        
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    
    # Spatial Encoding (Shortest Path Distance)
    print("Computing Shortest Path Distances...")
    spd_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if j in lengths.get(i, {}):
                    spd_matrix[i, j] = lengths[i][j]
                else:
                    spd_matrix[i, j] = 510 # max distance cutoff placeholder
    except Exception as e:
        print("Warning: SPD computation failed", e)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.spatial_encoding = spd_matrix
    return data

def main(input_dir="data/processed/ma_data", output_dir="data/processed/encodings"):
    os.makedirs(output_dir, exist_ok=True)
    
    for time_horizon in ['04h', '16h']:
        horizon_path = os.path.join(input_dir, time_horizon)
        if not os.path.exists(horizon_path):
            continue
            
        out_horizon_path = os.path.join(output_dir, time_horizon)
        os.makedirs(out_horizon_path, exist_ok=True)
        
        # Process Massachusetts
        ma_model = os.path.join(horizon_path, 'massachusetts_model.json')
        ma_res = os.path.join(horizon_path, 'massachusetts_ac_results.json')
        if os.path.exists(ma_model):
            print(f"Building graph for {ma_model}")
            data = build_graph(ma_model, ma_res)
            out_file = os.path.join(out_horizon_path, 'massachusetts.pt')
            torch.save(data, out_file)
            print(f"Saved {out_file}")

        # Process New England
        ne_model = os.path.join(horizon_path, 'new_england_model.json')
        ne_res = os.path.join(horizon_path, 'new_england_ac_results.json')
        if os.path.exists(ne_model):
            print(f"Building graph for {ne_model}")
            data = build_graph(ne_model, ne_res)
            out_file = os.path.join(out_horizon_path, 'new_england.pt')
            torch.save(data, out_file)
            print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
