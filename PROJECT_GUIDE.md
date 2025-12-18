# Traffic Flow Prediction Project Guide

This document explains the "Deep Learning Traffic Prediction" project, its file structure, features, and how to run it.

## 1. Project Overview
This project uses a **Spatio-Temporal Graph Neural Network (ST-GNN)** to predict traffic speeds on a highway network (PEMS-BAY dataset). 

**Key Features:**
- **Graph Neural Network**: Models the road network as a graph where sensors are nodes and roads are edges.
- **Spatio-Temporal Modeling**: Captures both spatial dependencies (connected roads affect each other) and temporal patterns (traffic flow over time).
- **Interactive Dashboard**: A Streamlit web app to visualize data and model performance.
- **[NEW] Video Analysis**: A feature to upload specialized traffic videos, extract speed data using Computer Vision (Optical Flow), and predict future traffic flow for that location.

## 2. File Explanations

### Core Application
- **`app.py`**: The main entry point for the **Streamlit Web Dashboard**. 
    - It handles the user interface (tabs for Dashboard, Predictions, Video Analysis).
    - Loads the trained model and dataset.
    - Manages the logic for the new "Video Analysis" tab, including file upload, saving to temp, and calling `video_utils`.

- **`video_utils.py`**: A helper module for **Computer Vision** tasks.
    - **`process_video_for_traffic_speed(video_path)`**: This function reads a video file frame-by-frame.
    - It uses **Optical Flow (Farneback algorithm)** to calculate how fast pixels are moving in the video.
    - It resizes frames to 320px for speed (Optimization).
    - It converts this pixel movement into a "speed" signal (MPH) that the AI model can understand.

### Model & Data
- **`st_gnn.py`**: Defines the **Neural Network Architecture**.
    - `GraphConvolution`: A layer that learns how traffic on one road affects its neighbors.
    - `TemporalConvNet`: A layer that learns patterns over time (e.g., rush hour trends).
    - `STGNN`: The main model combining these layers.

- **`data_loader.py`**: Handles **Data Preprocessing**.
    - Loads the PEMS-BAY dataset (`pems-bay.h5`) and the road graph (`adj_mx_bay.pkl`).
    - Normalizes the data (scales it so the AI can learn easily).
    - Splits data into Training, Validation, and Test sets.

- **`best_st_gnn.pth`**: The **Saved Model Weights**. This file contains the "brain" of the AI after it has been trained.

### Utility Scripts
- **`visualize.py`**: Helper functions to create the matplotlib charts and graphs seen in the app.
- **`train.py`**: The script used to train the model from scratch (not needed for running the dashboard).

## 3. How to Run

### Prerequisites
Ensure you have Python installed and the required libraries:
```bash
pip install -r requirements.txt
```
*(Key libraries: streamlit, torch, opencv-python, numpy, pandas, matplotlib)*

### command
To start the application, open your terminal in the project directory and run:

```bash
streamlit run app.py
```

### Troubleshooting
- **Port Error**: If it says "Port 8501 is already in use", you can specify a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Video Library Error**: If you see an error about `cv2`, run `pip install opencv-python`.
- **Command Not Found**: If `streamlit` is not recognized, try running it as a Python module:
  ```bash
  python -m streamlit run app.py
  ```

## 4. Expected Output

When you run the app, your browser will open to a dashboard with 5 tabs:

1.  **📊 Dashboard**: Shows overall model performance metrics (MAE, RMSE) and dataset statistics.
2.  **🔮 Predictions**: Allows you to pick a random sensor and see a graph of "Past Speed" vs "Predicted Speed" vs "Actual Speed".
3.  **📉 Error Analysis**: Shows how the model's accuracy changes for predictions further in the future.
4.  **🕸️ Spatial Graph**: Visualizes the connections between road sensors (Adjacency Matrix).
5.  **🎥 Video Analysis**:
    - **Input**: You upload a `.mp4` video of traffic.
    - **Process**: The app analyzes the video movement.
    - **Output**: It displays a graph showing the traffic speed extracted from your video (Blue line) and the AI's prediction for what happens next (Red line).
