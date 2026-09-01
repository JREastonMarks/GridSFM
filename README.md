# Work In Progress
**Note:** This is a work in progress and is not complete. It is subject to change at any time and may not work for your system.

# GridSFM: Regional Graphormer N-1 Grid Simulator

A specialized Graphormer model trained on the GridSFM dataset, isolated strictly to the Massachusetts/New England region for fast N-1 contingency testing.

## Overview
This project uses a Graphormer (Graph Transformer) architecture to predict cascading thermal and voltage violations in power grids under stress (e.g., node or edge failures). By isolating the GridSFM dataset to specific regional topologies and pre-computing spatial/edge encodings, the data pipeline fully bypasses CPU-bound data loading bottlenecks, enabling incredibly fast training cycles and real-time interactive inferences.

## Tech Stack & Environment
- **Language**: Python
- **Core Frameworks**: PyTorch, PyTorch Geometric, Microsoft GridSFM API
- **UI/Visualization**: Dash, Dash Cytoscape, Plotly
- **Hardware Support**: Apple Silicon MPS, CUDA, and CPU fallback support. Mixed precision training (`torch.amp.autocast`) is enabled by default.

---

## Setup Instructions

Ensure your Python environment has `torch`, `torch_geometric`, `networkx`, `dash`, `dash_cytoscape`, and `plotly` installed.

### 1. Data Pipeline & Preparation
The first step is downloading the raw Microsoft GridSFM dataset and processing it down to the regional topologies required. Run the following scripts in order:

```bash
# 1. Fetch the raw GridSFM data
python scripts/fetch_data.py

# 2. Filter the national grid down to Massachusetts and New England topologies
python scripts/filter_data.py

# 3. Pre-compute Spatial/Edge/Shortest-Path encodings and save to local SSD
python scripts/precompute_encodings.py
```
*Note: Pre-computing the encodings is crucial as it dramatically speeds up the training loop.*

### 2. Training the Model
To begin training the Regional Graphormer model, execute the training script. It will automatically detect hardware acceleration (MPS/CUDA).

```bash
export PYTHONPATH=.
python src/train.py
```
Checkpoints (including the normalization stats required for inference) will be saved into the `checkpoints/` directory as `best_model.pth`.

### 3. Running the UI Simulator
Once your data is encoded and your model is trained, you can launch the interactive N-1 Contingency Simulator UI.

```bash
export PYTHONPATH=.
python scripts/app.py
```
- Navigate to `http://127.0.0.1:8050` in your web browser.
- **Interactions**: Click on any node or transmission line in the grid map to intentionally disable it and trigger a cascade. 
- **Time-Series Player**: Drag the slider at the bottom of the screen to view predicted future states and thermal flows.
- **Region Switching**: Use the dropdown in the UI to switch between the Mass. and full New England topologies.

---

## Post-Processing & Analytics
The project also includes an `AdvancedGridAnalytics` engine (`src/analytics.py`) and a `GridDecoder` (`src/decoder.py`). These components translate the raw, normalized output tensors back into real-world physical values (Megawatts, Volts) for accurate feasibility scoring and violation counting.
