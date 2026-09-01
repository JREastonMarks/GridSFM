import json

base_data_dict = {'massachusetts': 1, 'new_england': 6}

with open('data/us-states.json') as f:
    us_states = json.load(f)

region_states = {
    'massachusetts': ['Massachusetts'],
    'new_england': ['Massachusetts', 'Connecticut', 'Rhode Island', 'Maine', 'New Hampshire', 'Vermont']
}

border_elements_dict = {'massachusetts': [], 'new_england': []}

for region in base_data_dict.keys():
    target_states = region_states[region]
    features = [f for f in us_states['features'] if f['properties']['name'] in target_states]
    
    # We will use dummy min_lat, max_lat, etc. for test
    min_lat, max_lat = 41.0, 43.0
    min_lon, max_lon = -73.0, -70.0
    
    def latlon_to_xy(lat, lon, min_l, max_l, min_lo, max_lo):
        x = (lon - min_lo) / (max_lo - min_lo + 1e-6) * 1000
        y = (max_l - lat) / (max_l - min_l + 1e-6) * 800
        return {'x': x, 'y': y}

    elems = []
    border_node_idx = 100000 # ensure no clash with grid nodes
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
                node_id = f"b_{border_node_idx}"
                elems.append({
                    'data': {'id': node_id},
                    'position': pos,
                    'locked': True,
                    'classes': 'border-node'
                })
                loop_nodes.append(node_id)
                border_node_idx += 1
                
            for i in range(len(loop_nodes) - 1):
                elems.append({
                    'data': {
                        'source': loop_nodes[i],
                        'target': loop_nodes[i+1],
                        'id': f"e_b_{loop_nodes[i]}_{loop_nodes[i+1]}"
                    },
                    'classes': 'border-edge'
                })
                
    border_elements_dict[region] = elems
    print(f"{region}: added {len(elems)} border elements")

