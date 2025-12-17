import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import your modules
# from data_loader import PEMSBAYDataLoader
# from st_gnn import STGNN


def visualize_predictions(model, test_loader, adj_mx, data_loader, device, 
                          num_samples=5, save_path='predictions.png'):
    """
    Visualize model predictions vs ground truth for multiple sensors
    
    Args:
        model: Trained ST-GNN model
        test_loader: Test data loader
        adj_mx: Adjacency matrix
        data_loader: Data loader object (for inverse transform)
        device: torch device
        num_samples: Number of time series to visualize
        save_path: Path to save figure
    """
    model.eval()
    adj_mx = adj_mx.to(device)
    
    # Get one batch
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(batch_x, adj_mx)
    
    # Move to CPU and convert to numpy
    batch_x = batch_x.cpu().numpy()
    batch_y = batch_y.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Select random sensors to visualize
    num_nodes = batch_x.shape[2]
    selected_sensors = np.random.choice(num_nodes, size=num_samples, replace=False)
    
    # Create figure
    fig = plt.figure(figsize=(15, 3 * num_samples))
    gs = GridSpec(num_samples, 1, figure=fig, hspace=0.4)
    
    for i, sensor_idx in enumerate(selected_sensors):
        ax = fig.add_subplot(gs[i, 0])
        
        # Get data for this sensor
        input_seq = batch_x[0, :, sensor_idx]  # First sample
        true_future = batch_y[0, :, sensor_idx]
        pred_future = predictions[0, :, sensor_idx]
        
        # Inverse transform (denormalize)
        # For simplicity, we'll plot normalized values
        # You can add inverse_transform here if needed
        
        # Time steps
        input_time = np.arange(len(input_seq))
        future_time = np.arange(len(input_seq), len(input_seq) + len(true_future))
        
        # Plot
        ax.plot(input_time, input_seq, 'b-', label='Input Sequence', linewidth=2)
        ax.plot(future_time, true_future, 'g-', label='Ground Truth', linewidth=2)
        ax.plot(future_time, pred_future, 'r--', label='Prediction', linewidth=2)
        
        # Vertical line separating input and prediction
        ax.axvline(x=len(input_seq)-0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Styling
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Traffic Speed (normalized)')
        ax.set_title(f'Sensor {sensor_idx} - Prediction vs Ground Truth')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Predictions visualization saved to {save_path}")
    plt.show()


def visualize_error_distribution(model, test_loader, adj_mx, device, save_path='error_dist.png'):
    """
    Visualize error distribution across all predictions
    
    Args:
        model: Trained model
        test_loader: Test data loader
        adj_mx: Adjacency matrix
        device: torch device
        save_path: Path to save figure
    """
    model.eval()
    adj_mx = adj_mx.to(device)
    
    all_errors = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x, adj_mx)
            
            # Compute absolute errors
            errors = torch.abs(predictions - batch_y)
            all_errors.append(errors.cpu().numpy().flatten())
    
    all_errors = np.concatenate(all_errors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(all_errors, vert=True)
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Error Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Error distribution saved to {save_path}")
    plt.show()


def visualize_spatial_attention(adj_mx, num_nodes_to_show=50, save_path='spatial_graph.png'):
    """
    Visualize the adjacency matrix (spatial connections)
    
    Args:
        adj_mx: Adjacency matrix
        num_nodes_to_show: Number of nodes to display (for readability)
        save_path: Path to save figure
    """
    # Convert to numpy
    adj_np = adj_mx.cpu().numpy() if torch.is_tensor(adj_mx) else adj_mx
    
    # Take subset for visualization
    adj_subset = adj_np[:num_nodes_to_show, :num_nodes_to_show]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_subset, cmap='YlOrRd', cbar=True, square=True)
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.title(f'Adjacency Matrix (First {num_nodes_to_show} nodes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Spatial graph saved to {save_path}")
    plt.show()


def analyze_error_by_time(model, test_loader, adj_mx, device, pred_len, save_path='error_by_time.png'):
    """
    Analyze how prediction error changes across prediction horizon
    
    Args:
        model: Trained model
        test_loader: Test data loader
        adj_mx: Adjacency matrix
        device: torch device
        pred_len: Prediction length
        save_path: Path to save figure
    """
    model.eval()
    adj_mx = adj_mx.to(device)
    
    # Store errors for each time step
    errors_by_step = [[] for _ in range(pred_len)]
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x, adj_mx)
            
            # Compute errors for each prediction step
            for t in range(pred_len):
                error = torch.abs(predictions[:, t, :] - batch_y[:, t, :])
                errors_by_step[t].append(error.cpu().numpy().flatten())
    
    # Compute average error for each step
    avg_errors = [np.mean(np.concatenate(errors)) for errors in errors_by_step]
    std_errors = [np.std(np.concatenate(errors)) for errors in errors_by_step]
    
    # Plot
    time_steps = np.arange(1, pred_len + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, avg_errors, 'b-o', linewidth=2, markersize=8)
    plt.fill_between(time_steps, 
                     np.array(avg_errors) - np.array(std_errors),
                     np.array(avg_errors) + np.array(std_errors),
                     alpha=0.3)
    
    plt.xlabel('Prediction Step (5-min intervals)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error vs Time Horizon')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Error by time analysis saved to {save_path}")
    plt.show()
    
    # Print statistics
    print("\nError by prediction step:")
    for i, (mae, std) in enumerate(zip(avg_errors, std_errors)):
        print(f"  Step {i+1}: MAE = {mae:.4f} ± {std:.4f}")


def create_prediction_report(model, test_loader, adj_mx, data_loader, device, pred_len):
    """
    Generate comprehensive prediction report with all visualizations
    
    Args:
        model: Trained model
        test_loader: Test data loader
        adj_mx: Adjacency matrix
        data_loader: Data loader object
        device: torch device
        pred_len: Prediction length
    """
    print("\n" + "="*50)
    print("GENERATING PREDICTION REPORT")
    print("="*50)
    
    # 1. Visualize predictions
    print("\n[1/4] Creating prediction visualizations...")
    visualize_predictions(model, test_loader, adj_mx, data_loader, device, 
                         num_samples=5, save_path='predictions.png')
    
    # 2. Error distribution
    print("[2/4] Analyzing error distribution...")
    visualize_error_distribution(model, test_loader, adj_mx, device, 
                                save_path='error_distribution.png')
    
    # 3. Spatial graph
    print("[3/4] Visualizing spatial connections...")
    visualize_spatial_attention(adj_mx, num_nodes_to_show=50, 
                               save_path='spatial_graph.png')
    
    # 4. Error by time
    print("[4/4] Analyzing error across prediction horizon...")
    analyze_error_by_time(model, test_loader, adj_mx, device, pred_len,
                         save_path='error_by_time.png')
    
    print("\n" + "="*50)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*50)
    print("\nGenerated files:")
    print("  - predictions.png")
    print("  - error_distribution.png")
    print("  - spatial_graph.png")
    print("  - error_by_time.png")


# Main script
if __name__ == "__main__":
    from data_loader import PEMSBAYDataLoader
    from st_gnn import STGNN
    
    # Configuration
    DATA_DIR = r"E:\deeplearning project\dataset"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'best_st_gnn.pth'
    
    SEQ_LEN = 12
    PRED_LEN = 12
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    NUM_LAYERS = 3
    
    print("Loading data and model...")
    
    # Load data
    data_loader = PEMSBAYDataLoader(
        data_dir=DATA_DIR,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        train_ratio=0.7,
        val_ratio=0.1
    )
    
    _, _, test_loader = data_loader.get_dataloaders(batch_size=BATCH_SIZE)
    
    # Load model
    model = STGNN(
        num_nodes=data_loader.num_nodes,
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=1,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN
    )
    
    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    # Generate report
    create_prediction_report(
        model=model,
        test_loader=test_loader,
        adj_mx=data_loader.adj_mx,
        data_loader=data_loader,
        device=DEVICE,
        pred_len=PRED_LEN
    )