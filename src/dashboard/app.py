import json

import httpx
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

API_URL = "http://localhost:8000"
st.set_page_config(page_title="UAV-GNN Routing Dashboard", layout="wide")
st.title("UAV-GNN Link Quality & Routing Dashboard")
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL)
try:
    health = httpx.get(f"{api_url}/health").json()
    st.sidebar.info(f"API Connected | Model: {health.get('model_id')}")
except Exception:
    st.sidebar.warning("API Disconnected. Ensure server is running.")
st.markdown("### 1. Input Network Data")
st.write("Input JSON network definition (nodes and edges) or use the sample below.")
sample_data = {
    "nodes": [
        {"node_id": 0, "x": 0, "y": 0, "z": 50, "vx": 5, "vy": 0, "vz": 0, "speed": 5, "degree": 2},
        {"node_id": 1, "x": 100, "y": 0, "z": 50, "vx": -5, "vy": 0, "vz": 0, "speed": 5, "degree": 2},
        {"node_id": 2, "x": 50, "y": 50, "z": 60, "vx": 0, "vy": 5, "vz": 0, "speed": 5, "degree": 2},
    ],
    "edges": [
        {
            "src": 0,
            "dst": 1,
            "distance": 100,
            "rssi": -65,
            "snr": 20,
            "delay": 5,
            "packet_loss": 0.01,
            "relative_speed": 10,
            "throughput": 10,
        },
        {
            "src": 1,
            "dst": 2,
            "distance": 70,
            "rssi": -55,
            "snr": 25,
            "delay": 3,
            "packet_loss": 0.005,
            "relative_speed": 5,
            "throughput": 15,
        },
        {
            "src": 0,
            "dst": 2,
            "distance": 70,
            "rssi": -80,
            "snr": 10,
            "delay": 15,
            "packet_loss": 0.05,
            "relative_speed": 5,
            "throughput": 5,
        },
    ],
}
data_input = st.text_area("Network JSON", value=json.dumps(sample_data, indent=2), height=250)
if st.button("Predict"):
    try:
        req_data = json.loads(data_input)
        with st.spinner("Processing..."):
            res = httpx.post(f"{api_url}/predict", json=req_data)
            res.raise_for_status()
            predictions = res.json()["predictions"]
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Predictions Table")
            st.dataframe(predictions, use_container_width=True)
        with col2:
            st.subheader("Network Visualization")
            G = nx.Graph()
            pos = {}
            for n in req_data["nodes"]:
                G.add_node(n["node_id"])
                pos[n["node_id"]] = (n["x"], n["y"])
            edge_colors = []
            for pred in predictions:
                u, v = (pred["src"], pred["dst"])
                G.add_edge(u, v)
                edge_colors.append("#2e7d32" if pred["stable"] else "#c62828")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axis("off")
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="#e0e0e0",
                font_size=9,
                edge_color=edge_colors,
                width=1.5,
                node_size=400,
                ax=ax,
            )
            import matplotlib.patches as mpatches

            green_patch = mpatches.Patch(color="#2e7d32", label="Stable Link")
            red_patch = mpatches.Patch(color="#c62828", label="Unstable Link")
            plt.legend(
                handles=[green_patch, red_patch], loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False
            )
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")
