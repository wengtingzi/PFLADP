# Fisher-AdpClipDP-PFL

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview
**Fisher-AdpClipDP-PFL** implements an adaptive differential privacy mechanism for **personalized federated learning**, featuring dynamic gradient clipping thresholds that adapt based on real-time gradient distributions. Our approach enhances the privacy-utility tradeoffs, building upon the **FedDPA framework** ([Yang et al., NeurIPS 2021](https://arxiv.org/abs/2107.09645)) with several key improvements.

## Key Features
- 🛡️ **Adaptive Differential Privacy**: Dynamic gradient clipping thresholds based on the distribution of gradients, ensuring improved privacy-utility tradeoffs.
- 🧠 **Personalized Federated Learning**: Adapts the privacy mechanisms specifically for personalized learning on each client.
- 🖥️ **Efficient Training**: Optimized for large-scale federated learning environments with efficient privacy-preserving mechanisms.

## Installation

1. **Create and activate a conda environment**:
   ```bash
   conda create -n FedPy39 python=3.9
   conda activate FedPy39
   conda init
   source ~/.bashrc
   conda activate FedPy39
2. **Install the core dependencies:**
    ```bash
    pip install tensorflow==2.15.0 tensorflow-estimator==2.15.0 tensorflow-privacy==0.8.11 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tensorflow-probability==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install fedlab torch

## Running the Experiment
    ```bash
    python main.py

## Notes
1. **CIFAR-10** dataset is the default dataset used for the example configuration. Modify the dataset option in `dataset` if you're using a different dataset.
2. Customize the privacy settings based on your specific requirements for `epsilon`, `delta`, `clipping bounds`, and `noise multipliers`.
3. This version directs users to the `options.py` file for configuration details and highlights the key parameters to focus on.



