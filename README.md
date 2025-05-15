# Small-World Modular Neural Networks with Izhikevich Dynamics

This project simulates **small-world modular neural networks** composed of excitatory and inhibitory Izhikevich neurons. The aim is to study how the **rewiring probability `p`** between modules affects spiking behavior, connectivity structure, and mean firing rates. The simulation was developed from scratch in Python as part of the Computational Neurodynamics course at Imperial College London.

## Objective

To generate directed small-world networks of Izhikevich neurons with modular structure, simulate their dynamics over time, and analyze how changes in **rewiring probability `p`** influence:

- Neuronal firing activity (raster plots)
- Mean firing rates across modules
- The structure of the excitatory connectivity matrix

## Network Architecture

- **Total neurons**: 1000 excitatory (split into 8 modules) + 200 inhibitory
- **Connectivity**:
  - Each module: 1000 random excitatory-to-excitatory links
  - Rewiring between modules with probability `p`
  - Local excitatory-to-inhibitory and diffuse inhibitory connections
- **Neuron models**:
  - Excitatory neurons: regular spiking
  - Inhibitory neurons: fast spiking
- **Simulation**: 1000 ms time window with Poisson noise

## Visualization Outputs

For each rewiring probability `p ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}`:
- **Connectivity Matrix** (Excitatory neurons only)
- **Raster Plot** of spike activity
- **Mean Firing Rate** for each module, computed using a 50ms sliding window with 20ms shifts

Plots are saved in vector format for analysis and comparison.

## How to Run

```bash
# Requirements
python3
numpy
matplotlib

# Run main simulation
python ModularNetwork.py
```

The output PDFs will be saved automatically for each value of `p`.

## Repository Structure

```bash
.
├── ModularNetwork.py           # Main simulation script
├── iznetwork.py                # Provided IzNetwork simulator class
├── report.pdf                  # PDF with all plots and figures
└── README.md
```
