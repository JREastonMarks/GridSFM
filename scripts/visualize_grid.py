import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from src.model import RegionalGraphormer
from scripts.simulate_contingency import ContingencySimulator
from src.decoder import GridDecoder

def create_animation():
    os.makedirs("visualizations", exist_ok=True)
    
    device = torch.device('cpu') # For rendering, CPU is safer with matplotlib
    
    # Load best model checkpoint
    chkpt_path = "checkpoints/best_model.pth"
    if not os.path.exists(chkpt_path):
        print(f"Error: {chkpt_path} not found.")
        return
        
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    
    # Initialize Model
    model = RegionalGraphormer(
        num_node_features=5, 
        num_edge_features=3, 
        hidden_dim=128, 
        output_dim=2 
    ).to(device)
    model.load_state_dict(chkpt['model_state_dict'])
    model.eval()
    
    # Initialize Decoder
    decoder = GridDecoder(normalization_stats=chkpt['stats'], device=device)
    
    # Load Base Graph
    data_path = "data/processed/encodings/04h/massachusetts.pt"
    data = torch.load(data_path, weights_only=False).to(device)
    
    simulator = ContingencySimulator(data)
    
    # Extract Graph topology for NetworkX
    G = nx.Graph()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))
    
    # data.edge_index is [2, E]. We extract tuples
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    
    # Extract positions: index 4 is lon (x), index 3 is lat (y)
    pos = {i: (data.x[i, 4].item(), data.x[i, 3].item()) for i in range(num_nodes)}
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    
    frames = 40
    def update(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Scenario: Ramping up load multiplier from 1.0 to 3.0 over 40 frames
        load_factor = 1.0 + (frame * 0.05)
        sim_data = simulator.scale_load(load_factor)
        
        # Apply Z-score normalization for inference
        stats = chkpt['stats']
        x_norm = (sim_data.x - stats['x_mean'].to(device)) / (stats['x_std'].to(device) + 1e-8)
        edge_norm = (sim_data.edge_attr - stats['edge_mean'].to(device)) / (stats['edge_std'].to(device) + 1e-8)
        
        with torch.no_grad():
            outputs = model(x_norm, sim_data.edge_index, edge_norm, sim_data.spatial_encoding.to(device))
            
        va, vm = decoder.decode_voltage(outputs)
        
        vmin = sim_data.x[:, 2]
        vmax = sim_data.x[:, 1]
        
        # Check for physical limit violations based on model predictions
        v_viols = decoder.check_voltage_violations(vm, vmin, vmax)
        t_viols = decoder.check_thermal_violations(va, vm, sim_data.edge_index, sim_data.edge_attr)
        
        # Dynamic Styling
        node_colors = ['#ff3333' if v else '#00ccff' for v in v_viols]
        edge_colors = ['#ff3333' if t else '#444444' for t in t_viols]
        node_sizes = [50 if v else 20 for v in v_viols]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=1.5, alpha=0.6)
        
        ax.set_title(f"Manifold-NE Grid Contingency Test\nLoad Demand Scale: {load_factor:.2f}x", color='white', fontsize=14, pad=20)
        ax.set_axis_off()
        
    print("Generating animation frames (this may take a minute)...")
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    out_path = "visualizations/load_ramp_simulation.gif"
    anim.save(out_path, writer='pillow')
    print(f"Animation successfully saved to {out_path}")

if __name__ == "__main__":
    create_animation()
