# PFLADP
Personalized Federated Learning with Adaptive Differential Privacy（本科生科研项目）

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Our approach builds upon the **FedDPA framework** ([Yang et al., NeurIPS 2021](https://arxiv.org/abs/2107.09645)) with several key improvements.

**Fisher-AdpClipDP-PFL** implements an adaptive differential privacy mechanism for **personalized federated learning**, featuring dynamic gradient clipping thresholds that adapt based on real-time gradient distributions which enhances the privacy-utility tradeoffs. 

**Fisher-VAE-Adaclip-PFL** is a personalized federated learning framework that integrates adaptive differential privacy and global shared data generation into the FedDPA framework. This framework enhances model generalization and accelerates convergence while maintaining strong privacy guarantees.

## 🔑 Key Features

- 🧠 **Fisher-Based Personalized Federated Learning**  
  Built upon the FedDPA framework proposed by [Yang et al.](https://arxiv.org/abs/2107.09645) , this approach utilizes the Fisher Information Matrix to dynamically decompose model parameters into personalized and global subsets.

- 📦 **VAE-Enhanced Global Data Sharing**  
  A variational autoencoder (VAE) is trained on server-side data to generate shared data samples. These are distributed to clients and mixed with local datasets to assist global parameter training, improving generalization and convergence.

- 🛡️ **Adaptive Differential Privacy with Adaptive Gradient Quantile Clipping**  
  Instead of using a fixed gradient clipping threshold, this framework employs an adaptive mechanism that calculates per-round clipping bounds based on the quantiles of local gradient norms. This allows the clipping threshold to reflect the actual distribution of gradients in each round, avoiding excessive or insufficient clipping. As a result, it improves the signal-to-noise ratio under differential privacy, leading to better convergence and higher model utility—especially in heterogeneous or non-IID federated settings.


## Installation

1. **Create and activate a conda environment**:
   ```bash
   conda create -n FedPy python=3.9
   conda activate FedPy
   conda init
   source ~/.bashrc
   conda activate FedPy
2. **Install the core dependencies:**
    ```bash
    pip install tensorflow==2.15.0 tensorflow-estimator==2.15.0 tensorflow-privacy==0.8.11 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tensorflow-probability==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install fedlab torch

## Notes
1. **CIFAR-10** dataset is the default dataset used for the example configuration. Modify the dataset option in `dataset` if you're using a different dataset.
2. Customize the privacy settings based on your specific requirements for `epsilon`, `delta`, `clipping bounds`,`noise multipliers` and `dir_alpha`.
3. This version directs users to the `options.py` file for configuration details and highlights the key parameters to focus on.

## Running the Experiment

```bash
# Baseline: FedDPA original framework
python base.py

# Proposed: Our Fisher-AdpClipDP-PFL improved algorithm
python Fisher-AdpClipDP-PFL/main_ours.py

# Proposed: Our Fisher-VAE-Adaclip-PFL improved algorithm
python Fisher-VAE-Adaclip-PFL/main_ours.py
