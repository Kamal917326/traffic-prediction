# 🚦 Traffic Flow Prediction using ST-GNN

## Overview

This project implements a **Spatio-Temporal Graph Neural Network (ST-GNN)** to predict traffic speeds on the PEMS-BAY dataset. It leverages the graph structure of road networks and historical time-series data to forecast future traffic conditions with high accuracy.

The system includes a complete pipeline for training, evaluation, and an interactive **Streamlit Dashboard** for visualization and real-time analysis.

## 🌟 Key Features

-   **Deep Learning Model**: Custom ST-GNN architecture combining Graph Convolutional Networks (GCN) and Gated Recurrent Units (GRU).
-   **Multi-Horizon Forecasting**: Predicts traffic speeds for **5, 15, 30, and 60 minutes** into the future.
-   **Performance Metrics**: Detailed evaluation providing MAE, RMSE, and MAPE for each prediction horizon.
-   **Interactive Dashboard**:
    -   **📊 Dashboard**: View overall model performance metrics in a clear, formatted table.
    -   **💾 Dataset Viewer**: Explore raw traffic data with generated timestamps.
    -   **🔮 Predictions**: Visualize forecasts vs. ground truth for specific sensors.
    -   **📉 Error Analysis**: Analyze how prediction error grows over time.
    -   **🕸️ Spatial Graph**: Visualize the sensor adjacency matrix.
    -   **🎥 Video Analysis**: Upload traffic videos to extract speed signals and predict future flow.

## 🛠️ Tech Stack

-   **Python 3.8+**
-   **PyTorch**: Deep learning framework.
-   **PyTorch Geometric**: Graph neural network layers.
-   **Streamlit**: Web interface for the dashboard.
-   **Pandas & NumPy**: Data manipulation.
-   **Matplotlib & Seaborn**: Visualization.
-   **OpenCV**: Video processing.

---

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <repo-name>
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset Setup**
    The dataset is included in the repo but compressed.
    -   **Unzip** `dataset/pems-bay.zip` to extract `pems-bay.h5`.
    -   Ensure `dataset/` contains:
        -   `pems-bay.h5`
        -   `adj_mx_bay.pkl`

---

## 🏃‍♂️ Usage

### 1. Training the Model
To train the ST-GNN model from scratch:

```bash
python train.py
```

-   The script will train for the configured number of epochs.
-   It will save the best model weights to `best_st_gnn.pth`.
-   **New**: At the end of training, it generates a `metrics.json` file containing the detailed performance table.

### 2. Running the Dashboard
To launch the interactive user interface:

```bash
streamlit run app.py
```

-   Open the provided URL (usually `http://localhost:8501`) in your browser.
-   Use the sidebar to view configuration info.
-   Navigate through the tabs to explore data and predictions.

---

## ❓ Troubleshooting

If you encounter issues, check these common solutions:

### ❌ Error: "Model not found at best_st_gnn.pth"
-   **Reason**: You haven't trained the model yet.
-   **Fix**: Run `python train.py` first. This will generate the `.pth` file required by the dashboard.

### ❌ Error: "No training metrics found"
-   **Reason**: The dashboard looks for `metrics.json`, which is created at the *end* of training.
-   **Fix**: Ensure `train.py` completes fully. If you interrupted it, run it again or click "Run Live Evaluation" in the dashboard to generate metrics on the fly.

### ❌ Error: "Video utilities not found"
-   **Reason**: OpenCV might be missing or corrupted.
-   **Fix**: Run `pip install opencv-python`.

### ❌ Error regarding "dataset path" or "FileNotFoundError"
-   **Reason**: The script cannot find `pems-bay.h5`.
-   **Fix**:
    1.  Check the `DATA_DIR` variable in `data_loader.py` and `app.py`.
    2.  Ensure your folder structure looks like this:
        ```
        project_root/
        ├── app.py
        ├── train.py
        └── dataset/
            ├── pems-bay.h5
            └── adj_mx_bay.pkl
        ```

### ❌ CUDA / GPU Out of Memory
-   **Reason**: The batch size might be too large for your GPU.
-   **Fix**: Open `train.py` and reduce `batch_size` (e.g., from 64 to 32 or 16).
    ```python
    # In train.py
    train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=16)
    ```

---

## 📂 Project Structure

-   `train.py`: Main script for training and evaluating the model.
-   `app.py`: Streamlit application code.
-   `st_gnn.py`: Definition of the Spatio-Temporal Graph Neural Network class.
-   `data_loader.py`: Handles loading and preprocessing of PEMS-BAY data.
-   `video_utils.py`: Helper functions for processing video input.
-   `metrics.json`: Stores model performance data for the UI.