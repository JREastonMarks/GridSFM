# Project: Manifold-NE

## Objective
Train a specialized Graphormer model on the GridSFM dataset isolated strictly to the Massachusetts/New England region for fast N-1 contingency testing.

## Tech Stack & Environment
- Language: Python
- Frameworks: PyTorch (configured for Apple Silicon 'mps' acceleration), PyTorch Geometric, Microsoft GridSFM API
- Target System: M4 MacBook Pro (64GB Unified Memory)

## Requirements & Guardrails
1. Data Pipeline: Filter the incoming GridSFM-Open dataset down to Massachusetts / ISO-NE topologies.
2. Optimization: Pre-compute Spatial/Edge/Shortest-Path encodings onto the local SSD to completely bypass the CPU data-loading bottleneck during active training.
3. Mixed Precision: Enforce `torch.amp.autocast('mps')` for hardware-accelerated FP16 training loops.
4. Evaluation: A post-processing decoder script to translate raw output tensors back into real-world physical values (Megawatts, Volts, feasibility scores).
