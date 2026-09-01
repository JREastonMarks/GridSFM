import dash
from dash import dcc, html, Input, Output, State, MATCH, no_update
import dash_cytoscape as cyto
import torch
import plotly.graph_objects as go
import json
from src.model import RegionalGraphormer
from scripts.simulate_contingency import ContingencySimulator
from src.decoder import GridDecoder
from src.analytics import AdvancedGridAnalytics

app = dash.Dash(__name__)

# --- Load Model & Data ---
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
chkpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=False)

model = RegionalGraphormer(
    num_node_features=5, num_edge_features=3, hidden_dim=128, output_dim=2 
).to(device)
model.load_state_dict(chkpt['model_state_dict'])
model.eval()

decoder = GridDecoder(normalization_stats=chkpt['stats'], device=device)
base_data_dict = {
    'massachusetts': torch.load("data/processed/encodings/04h/massachusetts.pt", weights_only=False).to(device),
    'new_england': torch.load("data/processed/encodings/04h/new_england.pt", weights_only=False).to(device)
}
simulator_dict = {k: ContingencySimulator(v) for k, v in base_data_dict.items()}
analytics_engine = AdvancedGridAnalytics(model, decoder, chkpt['stats'], device)

elements_dict = {}
base_edges_dict = {}
node_types_dict = {}

for region, b_data in base_data_dict.items():
    lats, lons = b_data.x[:, 3].cpu().numpy(), b_data.x[:, 4].cpu().numpy()
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()

    node_degrees = torch.zeros(b_data.num_nodes, dtype=torch.int, device=device)
    for src in b_data.edge_index[0]:
        node_degrees[src] += 1

    n_types = []
    for i in range(b_data.num_nodes):
        deg = node_degrees[i].item()
        pd = b_data.x[i, 0].item()
        if deg >= 6:
            n_types.append("Major Transfer Hub")
        elif pd > 250:
            n_types.append("Industrial Substation")
        else:
            n_types.append("Local Substation")
    node_types_dict[region] = n_types

    def latlon_to_xy(lat, lon, min_l, max_l, min_lo, max_lo):
        x = (lon - min_lo) / (max_lo - min_lo + 1e-6) * 1000
        y = (max_l - lat) / (max_l - min_l + 1e-6) * 800
        return {'x': x, 'y': y}

    elems = []
    seen_positions = {}
    for i in range(b_data.num_nodes):
        pos = latlon_to_xy(b_data.x[i, 3].item(), b_data.x[i, 4].item(), min_lat, max_lat, min_lon, max_lon)
        key = f"{pos['x']:.1f}_{pos['y']:.1f}"
        if key in seen_positions:
            seen_positions[key] += 1
            offset = seen_positions[key] * 15
            pos['x'] += offset
            pos['y'] += offset
        else:
            seen_positions[key] = 0
            
        elems.append({
            'data': {'id': f"{region}_{i}", 'label': str(i)},
            'position': pos,
            'locked': True
        })
        
    b_edges = b_data.edge_index.t().tolist()
    base_edges_dict[region] = b_edges
    for idx, (src, dst) in enumerate(b_edges):
        if src >= dst: 
            continue
        elems.append({
            'data': {'source': f"{region}_{src}", 'target': f"{region}_{dst}", 'edge_idx': idx, 'id': f"e_{region}_{idx}"},
        })
        
    # Load geographic state borders as unclickable cytoscape elements
    try:
        with open('data/us-states.json') as f:
            us_states = json.load(f)
            
        region_states = {
            'massachusetts': ['Massachusetts'],
            'new_england': ['Massachusetts', 'Connecticut', 'Rhode Island', 'Maine', 'New Hampshire', 'Vermont']
        }
        target_states = region_states[region]
        features = [f for f in us_states['features'] if f['properties']['name'] in target_states]
        
        border_node_idx = 1000000
        for f in features:
            geom = f['geometry']
            coords = geom['coordinates']
            loops = []
            if geom['type'] == 'Polygon':
                loops = coords
            elif geom['type'] == 'MultiPolygon':
                for poly in coords:
                    loops.extend(poly)
                    
            for loop in loops:
                loop_nodes = []
                for point in loop:
                    lon, lat = point
                    pos = latlon_to_xy(lat, lon, min_lat, max_lat, min_lon, max_lon)
                    node_id = f"b_{region}_{border_node_idx}"
                    elems.append({
                        'data': {'id': node_id},
                        'position': pos,
                        'locked': True,
                        'classes': 'state-border-node'
                    })
                    loop_nodes.append(node_id)
                    border_node_idx += 1
                    
                for i in range(len(loop_nodes) - 1):
                    elems.append({
                        'data': {
                            'source': loop_nodes[i],
                            'target': loop_nodes[i+1],
                            'id': f"e_{region}_{loop_nodes[i]}_{loop_nodes[i+1]}"
                        },
                        'classes': 'state-border-edge'
                    })
    except Exception as e:
        print("Warning: Could not load us-states.json", e)

    elements_dict[region] = elems


app.layout = html.Div(
    style={'backgroundColor': '#111', 'color': 'white', 'padding': '20px', 'fontFamily': 'sans-serif'},
    children=[
        html.H1("Regional Graphormer N-1 Grid Simulator", style={'textAlign': 'center'}),
        
        cyto.Cytoscape(
            id='cytoscape-grid',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '750px', 'backgroundColor': '#0a0a0a', 'borderRadius': '10px'},
            stylesheet=[],
            elements=elements_dict['massachusetts']
        ),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Select Region: ", style={'color': 'white', 'fontFamily': 'sans-serif', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[
                            {'label': 'Massachusetts', 'value': 'massachusetts'},
                            {'label': 'New England (ISO-NE)', 'value': 'new_england'}
                        ],
                        value='massachusetts',
                        clearable=False,
                        style={'width': '200px', 'color': 'black', 'display': 'inline-block'}
                    ),
                    html.Label(" Visualization Mode: ", style={'color': 'white', 'fontFamily': 'sans-serif', 'marginLeft': '20px', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='vis-mode-dropdown',
                        options=[
                            {'label': 'Violations (Default)', 'value': 'violations'},
                            {'label': 'Demand Heatmap', 'value': 'demand'}
                        ],
                        value='violations',
                        clearable=False,
                        style={'width': '200px', 'color': 'black', 'display': 'inline-block'}
                    ),
                ], style={'marginBottom': '15px'}),
                html.Label("24-Hour Load Curve Clock (0 = Base Load, 24 = Peak Load)", style={'color': 'white', 'fontFamily': 'sans-serif'}),
                html.Div([
                    html.Button('Play', id='play-button', n_clicks=0, style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'borderRadius': '5px', 'border': 'none', 'backgroundColor': '#00ccff', 'color': 'black', 'fontWeight': 'bold'}),
                    html.Button('Reset', id='reset-button', n_clicks=0, style={'marginRight': '20px', 'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'borderRadius': '5px', 'border': 'none', 'backgroundColor': '#ff4444', 'color': 'white', 'fontWeight': 'bold'}),
                    html.Div(
                        dcc.Slider(
                            0, 24, 1, 
                            value=0, 
                            id='time-slider',
                            marks={i: {'label': f'{i}h', 'style': {'color': 'white'}} for i in range(0, 25, 4)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ), style={'flexGrow': 1}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}),
                dcc.Interval(id='clock-interval', interval=1500, disabled=True)
            ], style={'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px', 'marginBottom': '10px'}),
            
            html.P("Hover over elements for live stats. Click on a node or transmission line to toggle an outage.", style={'color': '#aaa', 'fontFamily': 'sans-serif'}),
            
            # Store for disabled edge and node indices
            dcc.Store(id='disabled-edges', data=[]),
            dcc.Store(id='disabled-nodes', data=[]),
            dcc.Store(id='sim-data-store', data={'nodes': {}, 'edges': {}}),
            dcc.Store(id='red-team-targets', data=[]),
            
            html.Div([
                # Floating panel for hover info
                html.Div(
                    id='violation-counter',
                    style={
                        'position': 'absolute', 'top': '20px', 'left': '20px', 
                        'backgroundColor': 'rgba(255,0,0,0.8)', 'color': 'white', 
                        'padding': '10px 15px', 'borderRadius': '10px', 'zIndex': 9999, 
                        'pointerEvents': 'none', 'border': '1px solid #444',
                        'fontFamily': 'sans-serif', 'fontWeight': 'bold', 'fontSize': '16px'
                    },
                    children="Violations: 0"
                ),
                
                html.Div(
                    id='hover-info', 
                    style={
                        'position': 'absolute', 'top': '20px', 'right': '20px', 
                        'backgroundColor': 'rgba(0,0,0,0.8)', 'color': 'white', 
                        'padding': '15px', 'borderRadius': '10px', 'zIndex': 9999, 
                        'pointerEvents': 'none', 'border': '1px solid #444',
                        'minWidth': '200px', 'fontFamily': 'sans-serif'
                    },
                    children=html.P("Hover over a node or edge...", style={'margin': 0})
                ),
                
                html.Div([
                    dcc.Graph(id='violation-timeseries', style={'height': '300px'})
                ], style={'marginTop': '20px', 'backgroundColor': '#222', 'borderRadius': '10px', 'padding': '10px'})
            ], style={'position': 'relative'})
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.H2("Advanced Analytics", style={'color': 'white', 'fontFamily': 'sans-serif'}),
            html.Div([
                html.H4("1. Renewable Siting", title="Finds the maximum safe MW load you can inject into a specific node before causing system-wide limits using binary search.", style={'color': '#00ccff', 'margin': '10px 0 5px 0', 'cursor': 'help'}),
                dcc.Input(id='site-node', type='number', placeholder='Node ID', style={'width': '45%', 'marginRight': '5%'}),
                html.Button('Run Analysis', id='btn-siting', n_clicks=0, style={'width': '50%', 'padding': '5px'}),
                
                html.H4("2. What-If Expansion", title="Simulate building a new transmission line and calculate the difference in system-wide congestion.", style={'color': '#00ccff', 'margin': '20px 0 5px 0', 'cursor': 'help'}),
                dcc.Input(id='exp-src', type='number', placeholder='Src Node', style={'width': '30%', 'marginRight': '5%'}),
                dcc.Input(id='exp-dst', type='number', placeholder='Dst Node', style={'width': '30%', 'marginRight': '5%'}),
                html.Button('Test Line', id='btn-expansion', n_clicks=0, style={'width': '30%', 'padding': '5px'}),
                
                html.H4("3. N-k Stress Test", title="Monte Carlo simulation of N-k simultaneous edge drops to find extreme weather tipping points.", style={'color': '#00ccff', 'margin': '20px 0 5px 0', 'cursor': 'help'}),
                dcc.Input(id='nk-k', type='number', placeholder='k drop', value=3, style={'width': '45%', 'marginRight': '5%'}),
                html.Button('Run Monte Carlo', id='btn-nk', n_clicks=0, style={'width': '50%', 'padding': '5px'}),
                
                html.H4("4. Real-Time SCED", title="Uses backpropagation (gradient descent) through the Graphormer to find an optimal load/generation dispatch that zeroes out violations.", style={'color': '#00ccff', 'margin': '20px 0 5px 0', 'cursor': 'help'}),
                html.Button('Optimize Dispatch', id='btn-sced', n_clicks=0, style={'width': '100%', 'padding': '5px'}),
                
                html.H4("5. Adversarial Red Team", title="Greedy heuristic search (arXiv:1211.0709v1) to identify the minimal cut-set of nodes/lines that will maximize cascading failures.", style={'color': '#00ccff', 'margin': '20px 0 5px 0', 'cursor': 'help'}),
                dcc.Input(id='red-removals', type='number', placeholder='Removals', value=2, style={'width': '45%', 'marginRight': '5%'}),
                html.Button('Find Weakness', id='btn-redteam', n_clicks=0, style={'width': '50%', 'padding': '5px'}),
                
            ], style={'padding': '15px', 'backgroundColor': '#222', 'borderRadius': '10px'}),
            
            html.Div([
                html.H3("Results", style={'color': 'white', 'marginTop': '20px'}),
                dcc.Loading(
                    id="loading-analytics",
                    type="circle",
                    children=html.Div(id="analytics-results", style={'color': '#00ffcc', 'whiteSpace': 'pre-wrap', 'backgroundColor': '#111', 'padding': '10px', 'borderRadius': '5px', 'minHeight': '100px'})
                ),
                html.Button('Apply Found Outage', id='btn-apply-redteam', style={'display': 'none', 'marginTop': '10px', 'backgroundColor': '#ff3333', 'color': 'white', 'padding': '10px', 'border': 'none', 'borderRadius': '5px', 'width': '100%', 'cursor': 'pointer'})
            ])
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%', 'fontFamily': 'sans-serif'})
    ]
)

@app.callback(
    Output('disabled-edges', 'data'),
    Output('disabled-nodes', 'data'),
    Input('cytoscape-grid', 'tapEdge'),
    Input('cytoscape-grid', 'tapNode'),
    Input('btn-apply-redteam', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('region-dropdown', 'value'),
    State('red-team-targets', 'data'),
    State('disabled-edges', 'data'),
    State('disabled-nodes', 'data'),
    prevent_initial_call=True
)
def toggle_elements(tap_edge, tap_node, btn_apply, btn_reset, region, targets, disabled_edges, disabled_nodes):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-button' or trigger_id == 'region-dropdown':
        return [], []
    
    if disabled_edges is None: disabled_edges = []
    if disabled_nodes is None: disabled_nodes = []
    
    if 'cytoscape-grid' in trigger_id:
        if tap_edge:
            edge_idx = tap_edge['data']['edge_idx']
            if edge_idx in disabled_edges: disabled_edges.remove(edge_idx)
            else: disabled_edges.append(edge_idx)
        if tap_node:
            node_id_str = str(tap_node['data']['id'])
            if 'b_' in node_id_str: pass # Ignore border node clicks
            else:
                node_id = int(node_id_str.split('_')[-1])
                if node_id in disabled_nodes: disabled_nodes.remove(node_id)
                else: disabled_nodes.append(node_id)
            
    elif trigger_id == 'btn-apply-redteam' and targets:
        edges_list = base_data_dict[region].edge_index.t().tolist()
        for t in targets:
            if t[0] == 'edge':
                src, dst = t[1]
                for idx, (s, d) in enumerate(edges_list):
                    if (s == src and d == dst) or (s == dst and d == src):
                        if idx not in disabled_edges:
                            disabled_edges.append(idx)
            elif t[0] == 'node':
                if t[1] not in disabled_nodes:
                    disabled_nodes.append(t[1])
                    
    return disabled_edges, disabled_nodes

@app.callback(
    Output('cytoscape-grid', 'elements'),
    Input('region-dropdown', 'value')
)
def render_cytoscape(region):
    return elements_dict[region]

@app.callback(
    Output('cytoscape-grid', 'stylesheet'),
    Output('sim-data-store', 'data'),
    Output('violation-counter', 'children'),
    Input('time-slider', 'value'),
    Input('disabled-edges', 'data'),
    Input('disabled-nodes', 'data'),
    Input('region-dropdown', 'value'),
    Input('vis-mode-dropdown', 'value')
)
def update_grid_styles(time_val, disabled_edges, disabled_nodes, region, vis_mode):
    if disabled_edges is None: disabled_edges = []
    if disabled_nodes is None: disabled_nodes = []
    
    b_data = base_data_dict[region]
    simulator = simulator_dict[region]
    n_types = node_types_dict[region]
    b_edges = base_edges_dict[region]
    
    load_factor = 1.0 + (time_val / 24.0) * 2.5
    sim_data = simulator.scale_load(load_factor)
    
    node_active = torch.ones(sim_data.num_nodes, dtype=torch.bool, device=device)
    for n in disabled_nodes:
        node_active[n] = False
        sim_data.x[n, 0] = 0.0
        
    if disabled_edges or disabled_nodes:
        mask = torch.ones(sim_data.edge_index.size(1), dtype=torch.bool, device=device)
        for i in range(sim_data.edge_index.size(1)):
            src, dst = sim_data.edge_index[0, i].item(), sim_data.edge_index[1, i].item()
            if i in disabled_edges or not node_active[src] or not node_active[dst]:
                mask[i] = False
                rev_mask = (sim_data.edge_index[0] == dst) & (sim_data.edge_index[1] == src)
                mask[rev_mask] = False

        sim_data.edge_index = sim_data.edge_index[:, mask]
        sim_data.edge_attr = sim_data.edge_attr[mask]

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(sim_data.num_nodes))
    G.add_edges_from(sim_data.edge_index.t().tolist())
    
    in_main_grid = torch.ones(sim_data.num_nodes, dtype=torch.bool, device=device)
    if len(G.nodes) > 0:
        connected_components = list(nx.connected_components(G))
        if connected_components:
            largest_cc = max(connected_components, key=len)
            in_main_grid = torch.zeros(sim_data.num_nodes, dtype=torch.bool, device=device)
            in_main_grid[list(largest_cc)] = True
            
            mask = in_main_grid[sim_data.edge_index[0]] & in_main_grid[sim_data.edge_index[1]]
            sim_data.edge_index = sim_data.edge_index[:, mask]
            sim_data.edge_attr = sim_data.edge_attr[mask]
            
    sim_data.x[~in_main_grid, 0] = 0.0

    # Run inference using analytics engine to properly compute spatial encodings
    v_viols, t_viols, va, vm = analytics_engine.run_inference(sim_data, recompute_spatial=True)
    
    # Build dynamic stylesheet instead of replacing elements
    stylesheet = [
        {'selector': 'node', 'style': {'width': 10, 'height': 10, 'background-color': '#00ccff'}},
        {'selector': 'edge', 'style': {'width': 2, 'line-color': '#444'}},
        {'selector': 'node.state-border-node', 'style': {'width': 3, 'height': 3, 'background-color': '#00ff00'}},
        {'selector': 'edge.state-border-edge', 'style': {'width': 3, 'line-color': '#ffffff', 'line-style': 'solid'}}
    ]
    
    sim_data_dict = {'nodes': {}, 'edges': {}}
    
    num_node_viols = 0
    num_edge_viols = 0
    
    if vis_mode == 'demand':
        max_load = float(sim_data.x[:, 0].max())
        if max_load == 0: max_load = 1.0

    for i in range(sim_data.num_nodes):
        is_viol = bool(v_viols[i])
        is_offline = i in disabled_nodes
        is_isolated = not in_main_grid[i].item()
        
        current_load = 0.0 if (is_offline or is_isolated) else float(sim_data.x[i, 0])
        
        if is_offline or is_isolated:
            stylesheet.append({'selector': f'[id = "{region}_{i}"]', 'style': {'background-color': '#444', 'width': 8, 'height': 8, 'opacity': 0.5}})
        elif vis_mode == 'demand':
            ratio = current_load / max_load
            r = int(255 * ratio)
            b = int(255 * (1 - ratio))
            size = 10 + 20 * ratio
            style_dict = {'background-color': f'rgb({r},0,{b})', 'width': size, 'height': size}
            if is_viol:
                style_dict['border-width'] = 2
                style_dict['border-color'] = '#ffff00'
                num_node_viols += 1
            stylesheet.append({'selector': f'[id = "{region}_{i}"]', 'style': style_dict})
        elif is_viol:
            stylesheet.append({'selector': f'[id = "{region}_{i}"]', 'style': {'background-color': '#ff3333', 'width': 20, 'height': 20}})
            num_node_viols += 1
            
        sim_data_dict['nodes'][str(i)] = {
            'type': n_types[i],
            'vm': float(vm[i]),
            'va': float(va[i]),
            'violation': is_viol,
            'offline': is_offline,
            'isolated': is_isolated,
            'base_load': float(b_data.x[i, 0]),
            'current_load': 0.0 if (is_offline or is_isolated) else float(sim_data.x[i, 0]),
            'vmax': float(b_data.x[i, 1]),
            'vmin': float(b_data.x[i, 2])
        }
        
    sim_edge_tuples = set(map(tuple, sim_data.edge_index.t().tolist()))
    viol_dict = {tuple(sim_data.edge_index[:, j].tolist()): bool(t_viols[j]) for j in range(len(t_viols))}
    
    for idx, (src, dst) in enumerate(b_edges):
        if src >= dst: 
            continue
            
        is_disabled = tuple([src,dst]) not in sim_edge_tuples and tuple([dst,src]) not in sim_edge_tuples
        is_overload = viol_dict.get((src, dst), False) or viol_dict.get((dst, src), False)
        
        rate = float(b_data.edge_attr[idx, 2].item())
        status = "ONLINE"
        
        if is_disabled:
            stylesheet.append({'selector': f'[id = "e_{region}_{idx}"]', 'style': {'line-color': '#222', 'line-style': 'dashed', 'width': 1}})
            status = "OFFLINE"
        elif is_overload:
            stylesheet.append({'selector': f'[id = "e_{region}_{idx}"]', 'style': {'line-color': '#ff3333', 'width': 5}})
            status = "OVERLOAD"
            num_edge_viols += 1
            
        sim_data_dict['edges'][str(idx)] = {
            'source': src,
            'target': dst,
            'rate': rate,
            'status': status
        }
        
    total_viols = num_node_viols + num_edge_viols
    counter_text = f"Violations: {total_viols} ({num_node_viols} Node, {num_edge_viols} Line)"
        
    return stylesheet, sim_data_dict, counter_text

@app.callback(
    Output('hover-info', 'children'),
    Input('cytoscape-grid', 'mouseoverNodeData'),
    Input('cytoscape-grid', 'mouseoverEdgeData'),
    State('sim-data-store', 'data')
)
def display_hover(hover_node, hover_edge, sim_data):
    ctx = dash.callback_context
    if not ctx.triggered or not sim_data: 
        return html.P("Hover over a node or edge...", style={'margin': 0})
        
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'mouseoverNodeData' in trigger_id and hover_node:
        node_id_full = str(hover_node.get('id', ''))
        node_id = node_id_full.split('_')[-1] if '_' in node_id_full else node_id_full
        data = sim_data.get('nodes', {}).get(node_id, {})
        
        status_text = "Normal"
        color = '#00ccff'
        if data.get('offline'):
            status_text = "OFFLINE (USER DISABLED)"
            color = '#888'
        elif data.get('isolated'):
            status_text = "ISOLATED (0 LOAD)"
            color = '#888'
        elif data.get('violation'):
            status_text = "⚠️ VOLTAGE VIOLATION"
            color = '#ff3333'
            
        node_type = data.get('type', 'Unknown')
            
        return html.Div([
            html.H3(f"Node {node_id} - {node_type}", style={'margin': '0 0 10px 0', 'color': color, 'fontSize': '18px'}),
            html.P(f"Current Load: {data.get('current_load', 0):.2f} MW (Base: {data.get('base_load', 0):.2f})", style={'margin': '5px 0'}),
            html.P(f"Voltage Limits: {data.get('vmin', 0):.2f} - {data.get('vmax', 0):.2f} p.u.", style={'margin': '5px 0'}),
            html.P(f"Simulated Voltage (Vm): {data.get('vm', 0):.3f} p.u.", style={'margin': '5px 0'}),
            html.P(f"Status: {status_text}", style={'margin': '10px 0 0 0', 'fontWeight': 'bold', 'color': color})
        ])
        
    elif 'mouseoverEdgeData' in trigger_id and hover_edge:
        edge_idx = str(hover_edge.get('edge_idx', ''))
        data = sim_data.get('edges', {}).get(edge_idx, {})
        status = data.get('status', 'UNKNOWN')
        color = '#ff3333' if status == 'OVERLOAD' else ('#aaa' if status == 'OFFLINE' else '#00ccff')
        return html.Div([
            html.H3(f"Line {data.get('source')}-{data.get('target')}", style={'margin': '0 0 10px 0', 'color': color}),
            html.P(f"Max Rate: {data.get('rate', 0):.2f}", style={'margin': '5px 0'}),
            html.P(f"Status: {status}", style={'margin': '10px 0 0 0', 'fontWeight': 'bold', 'color': color})
        ])
        
    return html.P("Hover over a node or edge...", style={'margin': 0})

@app.callback(
    Output('clock-interval', 'disabled'),
    Output('play-button', 'children'),
    Input('play-button', 'n_clicks'),
    State('clock-interval', 'disabled'),
    prevent_initial_call=True
)
def toggle_play(n_clicks, disabled):
    if disabled:
        return False, "Pause"
    else:
        return True, "Play"

@app.callback(
    Output('time-slider', 'value'),
    Input('clock-interval', 'n_intervals'),
    State('time-slider', 'value'),
    prevent_initial_call=True
)
def tick(n, current_val):
    if current_val is None: current_val = 0
    return (current_val + 1) % 25

@app.callback(
    Output('analytics-results', 'children'),
    Output('red-team-targets', 'data'),
    Output('btn-apply-redteam', 'style'),
    Input('btn-siting', 'n_clicks'),
    Input('btn-expansion', 'n_clicks'),
    Input('btn-nk', 'n_clicks'),
    Input('btn-sced', 'n_clicks'),
    Input('btn-redteam', 'n_clicks'),
    State('site-node', 'value'),
    State('exp-src', 'value'),
    State('exp-dst', 'value'),
    State('nk-k', 'value'),
    State('red-removals', 'value'),
    State('region-dropdown', 'value'),
    prevent_initial_call=True
)
def run_analytics(btn_site, btn_exp, btn_nk, btn_sced, btn_red, site_node, exp_src, exp_dst, nk_k, red_removals, region):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    hide_btn = {'display': 'none', 'marginTop': '10px', 'backgroundColor': '#ff3333', 'color': 'white', 'padding': '10px', 'border': 'none', 'borderRadius': '5px', 'width': '100%', 'cursor': 'pointer'}
    show_btn = {'display': 'block', 'marginTop': '10px', 'backgroundColor': '#ff3333', 'color': 'white', 'padding': '10px', 'border': 'none', 'borderRadius': '5px', 'width': '100%', 'cursor': 'pointer', 'fontWeight': 'bold'}
    
    b_data = base_data_dict[region]
    
    try:
        if trigger_id == 'btn-siting':
            if site_node is None: return "Please enter a Node ID.", [], hide_btn
            cap = analytics_engine.hosting_capacity(b_data, int(site_node))
            return f"Optimal Siting:\nNode {site_node} can host up to {cap:.2f} MW of new generation before causing system-wide thermal or voltage violations.", [], hide_btn
            
        elif trigger_id == 'btn-expansion':
            if exp_src is None or exp_dst is None: return "Please enter Src and Dst nodes.", [], hide_btn
            base_v, new_v = analytics_engine.expansion_planning(b_data, int(exp_src), int(exp_dst))
            return f"What-If Expansion ({exp_src} -> {exp_dst}):\nBase Violations: {base_v}\nViolations after new line: {new_v}\nDelta: {new_v - base_v}", [], hide_btn
            
        elif trigger_id == 'btn-nk':
            if nk_k is None: nk_k = 3
            failure_rate, worst, max_v = analytics_engine.nk_contingency_mc(b_data, num_samples=50, k=int(nk_k))
            res = f"N-{nk_k} Monte Carlo (50 runs):\nSystem Failure Rate: {failure_rate:.1f}%\n"
            targets = []
            if worst:
                res += f"Worst combination found:\n{worst}\n(Caused {max_v} total violations)"
                targets = [['edge', w] for w in worst]
            return res, targets, show_btn
            
        elif trigger_id == 'btn-sced':
            v_viols, t_viols, _, _ = analytics_engine.run_inference(b_data)
            v_c, t_c = analytics_engine.count_violations(b_data, v_viols, t_viols)
            base_viols = v_c + t_c
            optimized_x = analytics_engine.run_sced(b_data, lr=10.0, steps=100)
            
            temp_data = b_data.clone()
            temp_data.x = optimized_x
            v_viols_new, t_viols_new, _, _ = analytics_engine.run_inference(temp_data)
            v_c_new, t_c_new = analytics_engine.count_violations(temp_data, v_viols_new, t_viols_new)
            new_viols = v_c_new + t_c_new
            
            return f"SCED Gradient Optimization:\nInitial Violations: {base_viols}\nPost-SCED Violations: {new_viols}\nDispatch successfully updated using backpropagation!", [], hide_btn
            
        elif trigger_id == 'btn-redteam':
            if red_removals is None: red_removals = 2
            worst_targets, max_v = analytics_engine.red_team_heuristic(b_data, max_removals=int(red_removals))
            res = f"Red Team Adversarial Search:\nFound {len(worst_targets)} critical weak points:\n{worst_targets}\n\nRemoving these guarantees {max_v} cascading thermal/voltage violations."
            return res, worst_targets, show_btn
            
    except Exception as e:
        return f"Error running analysis: {str(e)}", [], hide_btn
        
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('violation-timeseries', 'figure'),
    Input('disabled-edges', 'data'),
    Input('disabled-nodes', 'data'),
    Input('region-dropdown', 'value')
)
def update_timeseries(disabled_edges, disabled_nodes, region):
    if disabled_edges is None: disabled_edges = []
    if disabled_nodes is None: disabled_nodes = []
    
    simulator = simulator_dict[region]
    
    # Pre-calculate topology and spatial encoding ONCE
    sim_data_base = simulator.scale_load(1.0)
    node_active = torch.ones(sim_data_base.num_nodes, dtype=torch.bool, device=device)
    for n in disabled_nodes:
        node_active[n] = False
        
    mask = torch.ones(sim_data_base.edge_index.size(1), dtype=torch.bool, device=device)
    if disabled_edges or disabled_nodes:
        for i in range(sim_data_base.edge_index.size(1)):
            src, dst = sim_data_base.edge_index[0, i].item(), sim_data_base.edge_index[1, i].item()
            if i in disabled_edges or not node_active[src] or not node_active[dst]:
                mask[i] = False
                rev_mask = (sim_data_base.edge_index[0] == dst) & (sim_data_base.edge_index[1] == src)
                mask[rev_mask] = False
                
    sim_data_base.edge_index = sim_data_base.edge_index[:, mask]
    sim_data_base.edge_attr = sim_data_base.edge_attr[mask]
    
    # Apply isolation logic
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(sim_data_base.num_nodes))
    G.add_edges_from(sim_data_base.edge_index.t().tolist())
    
    in_main_grid = torch.ones(sim_data_base.num_nodes, dtype=torch.bool, device=device)
    if len(G.nodes) > 0:
        connected_components = list(nx.connected_components(G))
        if connected_components:
            largest_cc = max(connected_components, key=len)
            in_main_grid = torch.zeros(sim_data_base.num_nodes, dtype=torch.bool, device=device)
            in_main_grid[list(largest_cc)] = True
            
            mask_iso = in_main_grid[sim_data_base.edge_index[0]] & in_main_grid[sim_data_base.edge_index[1]]
            sim_data_base.edge_index = sim_data_base.edge_index[:, mask_iso]
            sim_data_base.edge_attr = sim_data_base.edge_attr[mask_iso]
            
    # Compute spatial encoding once for the base topology
    spatial_enc = analytics_engine.compute_spatial_encoding(sim_data_base)
    
    x_vals = list(range(25))
    y_v = []
    y_t = []
    
    for t in x_vals:
        load_factor = 1.0 + (t / 24.0) * 2.5
        sim_data_t = simulator.scale_load(load_factor)
        
        # Apply masks to node features
        for n in disabled_nodes:
            sim_data_t.x[n, 0] = 0.0
        sim_data_t.x[~in_main_grid, 0] = 0.0
        
        # Inject precomputed topology
        sim_data_t.edge_index = sim_data_base.edge_index
        sim_data_t.edge_attr = sim_data_base.edge_attr
        sim_data_t.spatial_encoding = spatial_enc
            
        v_viols, t_viols, _, _ = analytics_engine.run_inference(sim_data_t, recompute_spatial=False)
        v_c, t_c = analytics_engine.count_violations(sim_data_t, v_viols, t_viols)
        y_v.append(v_c)
        y_t.append(t_c)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_v, mode='lines+markers', name='Voltage Violations', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=x_vals, y=y_t, mode='lines+markers', name='Thermal Violations', line=dict(color='red')))
    
    fig.update_layout(
        title="Predicted Violations over 24-Hour Load Curve",
        plot_bgcolor='#222', paper_bgcolor='#222', font=dict(color='white'),
        xaxis=dict(title='Hour of Day'), yaxis=dict(title='Violation Count'),
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
