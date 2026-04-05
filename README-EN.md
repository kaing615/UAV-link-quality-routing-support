<p align="center">
  <a href="./README.md"><strong>[ 🇻🇳 Tiếng Việt ]</strong></a>
  &nbsp;&nbsp;&nbsp;
  <strong>[ 🇺🇸 English ]</strong>
</p>

<h1 align="center">UAV Link Quality Routing Support</h1>

<p align="center">
  Application of Graph Neural Networks for Link Quality Prediction and Routing Support in UAV Networks
</p>

<p align="center">
  <i>Ứng dụng Graph Neural Network trong dự đoán chất lượng liên kết và hỗ trợ định tuyến trong mạng UAV</i>
</p>

---

- **Project Title:** Application of Graph Neural Networks for Link Quality Prediction and Routing Support in UAV Networks
- **University:** University of Information Technology - Vietnam National University Ho Chi Minh City
- **Faculty:** Faculty of Computer Networks and Communications
- **Supervisor:** M.Sc. Dang Le Bao Chuong
- **Students:** Nguyen Dinh Tam - 23521389, Vo Cong Vinh - 23521800

---

## Introduction

This project focuses on the problem of **link quality prediction** and **routing support** in dynamic UAV networks using **Graph Neural Networks (GNNs)**. The core idea is to jointly exploit the **global network topology**, **node-level features**, and **edge-level features** to predict future link conditions, then use these predictions as a supporting signal for more stable route selection in highly dynamic UAV ad hoc networks.

In UAV ad hoc networks, unmanned aerial vehicles operate in **three-dimensional space**, continuously moving with changing speed and direction. As a result, the **network topology changes rapidly over time**, making wireless links highly unstable and prone to degradation or disconnection. This directly affects end-to-end communication performance.

Instead of relying only on instantaneous information such as **hop count**, **position**, or **distance**, this project aims to build a learning-based module capable of capturing the **spatial and structural relationships of UAV networks** through graph representation. Based on that representation, the model predicts the quality or state of wireless links in the next time step and then maps the prediction into a signal that supports the **route selection process**.

In other words, this project does **not aim to design a completely new routing protocol from scratch**. Its main objective is to develop a **link quality prediction module** that can assist routing decisions and improve path stability in dynamic UAV networks.

---

## Core Objectives

1. **Simulate dynamic UAV networks in 3D space:**
   Build a simulation environment where UAVs move in three-dimensional space and the network topology evolves over time.

2. **Collect topology snapshots at each time step:**
   Record the network state at each time step or observation window for machine learning data construction.

3. **Represent the network as a graph:**
   Model the UAV network as a graph, where UAVs are **nodes** and wireless links are **edges**.

4. **Construct node and edge feature sets:**
   Extract features such as node position, velocity, and degree, as well as edge distance, RSSI, SNR, delay, packet loss, and relative speed.

5. **Label link states:**
   Define link conditions such as **stable / degraded** or other quality levels for supervised learning.

6. **Train a GNN model for link prediction:**
   Use **GraphSAGE** as the main model to learn graph representations and predict edge-level link states.

7. **Compare with baselines and alternative models:**
   Evaluate performance against **Logistic Regression**, **Random Forest**, **MLP**, threshold-based methods, and **GAT**.

8. **Support routing using predicted outputs:**
   Map predicted scores or classes into **edge weights** or **priority metrics** to support more reliable route selection.

9. **Evaluate the system at two levels:**
   - **Machine learning level:** Accuracy, Precision, Recall, F1-score, ROC-AUC
   - **Network performance level:** Packet Delivery Ratio, End-to-End Delay, Route Stability, Throughput

---

## Core Idea

The overall pipeline of the project consists of the following steps:

1. **Simulate a dynamic UAV network** in 3D space
2. **Collect topology snapshots** at each time step
3. **Build a graph dataset** from topology and link-related features
4. **Train a GNN model** to predict the future state of each link
5. **Map the predicted output** into edge weights or priority criteria
6. **Support more stable route selection** in dynamic UAV networks

In short, the system acts as a **routing support module**, where the GNN provides predicted link quality information to help the routing layer make better decisions in a highly dynamic environment.

---

## Technology Stack

| Component                           | Technology                              |
| ----------------------------------- | --------------------------------------- |
| **Programming Language**            | Python                                  |
| **Simulation**                      | Python-based UAV simulator              |
| **Data Processing**                 | NumPy, Pandas                           |
| **Visualization**                   | Matplotlib                              |
| **Graph Representation / Learning** | PyTorch Geometric                       |
| **Baseline Machine Learning**       | Logistic Regression, Random Forest, MLP |
| **Heuristic Methods**               | Threshold-based methods                 |
| **Main Model**                      | GraphSAGE                               |
| **Comparison Model**                | GAT                                     |

---

## Input Features

The system jointly exploits information at two levels:

### 1. Node Features

- 3D position of UAVs
- Velocity / movement direction
- Node degree
- Local neighborhood information

### 2. Edge Features

- Distance between two UAVs
- RSSI
- SNR
- Delay
- Packet loss
- Relative speed between two UAVs

By combining **node features**, **edge features**, and **graph topology**, the model can capture not only the local condition of a specific link but also the **global structural context of the network**.

---

## Overall Workflow

1. **Simulation stage:**
   The system generates UAV movement scenarios in 3D space and produces the corresponding network topologies over time.

2. **Data collection stage:**
   At each time step, the system records node information, edge information, and link conditions to build graph snapshots.

3. **Dataset construction stage:**
   The snapshots are preprocessed, feature-engineered, and labeled to create the dataset used for training and evaluation.

4. **Model training stage:**
   The GNN learns graph representations and predicts the state or quality of links in the next time step.

5. **Routing support stage:**
   The predicted output is mapped into priority scores or edge weights to support more stable route selection.

6. **Evaluation stage:**
   The system is evaluated both in terms of prediction performance and its practical impact on network performance.

---

## Proposed Project Structure

```text
uav-link-quality-routing
├── data/                     # Raw data, processed data, graph snapshots
├── simulation/               # UAV simulation environment and topology generation
├── preprocessing/            # Data preprocessing and node/edge feature extraction
├── models/                   # GraphSAGE, GAT, and baseline implementations
├── training/                 # Training, validation, and testing scripts
├── routing/                  # Logic for mapping predictions into routing support
├── evaluation/               # Model evaluation and network-level assessment
├── utils/                    # Utility functions
├── configs/                  # Configuration files for simulation and training
├── outputs/                  # Logs, figures, checkpoints, and experiment results
└── README.md
```
