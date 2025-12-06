This repository contains a compact Python implementation of a LIF-based spiking neural network designed to explore:

- How **LIF neurons** integrate and fire in response to incoming spikes,
- How **pair-based STDP** can be used to update synaptic weights based on spike timing,
- How a simple **homeostatic / balancing mechanism** can stabilize network activity over time.

The goal is to provide a **clean, educational reference** implementation that can be extended into more complex SNN architectures or used as a starting point for course projects and research experiments.

---

## Features

- **Leaky Integrate-and-Fire neuron dynamics**
  - Membrane potential integration with leak.
  - Threshold-based spiking and reset.

- **STDP learning rule**
  - Synaptic weight updates driven by pre–post spike timing.
  - Supports experimentation with learning windows and parameters.

- **Balancing / homeostatic mechanism**
  - Basic mechanism for keeping activity in a reasonable range
    (e.g. through weight normalization or rate-based adjustment).

- **Single-file prototype**
  - All core logic is contained in `stdp+bm.py` for easy reading and modification.
