import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
print("Importing local modules...")
print("Importing data_loader...")
from data_loader import PEMSBAYDataLoader
print("Importing st_gnn...")
from st_gnn import STGNN
print("Imports done.")


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics
    
    Args:
        predictions: (batch, pred_len, num_nodes)
        targets: (batch, pred_len, num_nodes)
    
    Returns:
        mae, rmse, mape
    """
    # Mean Absolute Error
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Root Mean Squared Error
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    # Mean Absolute Percentage Error
    epsilon = 1e-5  # Avoid division by zero
    mape = torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100
    mape = mape.item()
    
    return mae, rmse, mape


class Trainer:
    """Training and evaluation pipeline"""
    
    def __init__(self, model, data_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        
        # Loss function (MAE is common for traffic prediction)
        self.criterion = nn.L1Loss()  # MAE
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_mape': []
        }
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, adj_mx):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        adj_mx = adj_mx.to(self.device)
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            # Move to device
            batch_x = batch_x.to(self.device)  # (batch, seq_len, num_nodes)
            batch_y = batch_y.to(self.device)  # (batch, pred_len, num_nodes)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x, adj_mx)
            
            # Compute loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, adj_mx):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        adj_mx = adj_mx.to(self.device)
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x, adj_mx)
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        mae, rmse, mape = compute_metrics(all_predictions, all_targets)
        
        return avg_loss, mae, rmse, mape
    
    def train(self, epochs, train_loader, val_loader, adj_mx, save_path='best_model.pth'):
        """
        Full training loop
        
        1. Iterates through epochs
        2. Trains the model (updates weights)
        3. Validates the model (checks performance on unseen data)
        4. Saves the best model based on validation loss
        
        Args:
            epochs: Number of training epochs (passes through the entire dataset)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            adj_mx: Adjacency matrix (graph structure)
            save_path: File path to save the best model weights
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # --- TRAIN STEP ---
            # Updates model weights using backpropagation
            train_loss = self.train_epoch(train_loader, adj_mx)
            
            # --- VALIDATION STEP ---
            # Evaluates model on data it hasn't seen during training
            # Used to tune hyperparameters and prevent overfitting
            val_loss, val_mae, val_rmse, val_mape = self.validate(val_loader, adj_mx)
            
            # --- SCHEDULER UPDATE ---
            # Reduces learning rate if validation loss stops improving
            # This helps the model converge to a better solution
            self.scheduler.step(val_loss)
            
            # Save history for plotting
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_mape'].append(val_mape)
            
            # --- MODEL CHECKPOINTING ---
            # We save the model only if the current validation loss is lower than the best seen so far.
            # This ensures we keep the "best" version of the model, not just the last one.
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                best_marker = " (★ Best)"
            else:
                best_marker = ""
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s){best_marker}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val MAE:    {val_mae:.4f}")
            print(f"  Val RMSE:   {val_rmse:.4f}")
            print(f"  Val MAPE:   {val_mape:.2f}%")
        
        total_time = time.time() - start_time
        print(f"\n✓ Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def test(self, test_loader, adj_mx, model_path='best_model.pth'):
        """
        Test the model on test set and print horizon-specific metrics
        """
        print("\nTesting model...")
        
        # Load best model
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        adj_mx = adj_mx.to(self.device)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_x, adj_mx)
                
                all_preds.append(predictions.cpu())
                all_targets.append(batch_y.cpu())
        
        # Concatenate
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Horizons to evaluate (indices assuming 5min steps)
        # 5min: step 1 (idx 0), 15min: step 3 (idx 2), 30min: step 6 (idx 5), 60min: step 12 (idx 11)
        horizons = [0, 2, 5, 11]
        metrics_results = {'MAE': [], 'RMSE': [], 'MAPE': []}
        
        for h in horizons:
            if h < all_preds.shape[1]:
                pred_h = all_preds[:, h, :]
                target_h = all_targets[:, h, :]
                mae, rmse, mape = compute_metrics(pred_h, target_h)
                metrics_results['MAE'].append(mae)
                metrics_results['RMSE'].append(rmse)
                metrics_results['MAPE'].append(mape)
            else:
                for k in metrics_results:
                    metrics_results[k].append(0.0)
        
        # Print table
        print("\n" + "="*65)
        print(f"{'Dataset':<10} {'Metric':<8} {'5min':<10} {'15min':<10} {'30min':<10} {'60min':<10}")
        print("-" * 65)
        
        # MAE
        mae_vals = metrics_results['MAE']
        print(f"{'PEMS-BAY':<10} {'MAE':<8} {mae_vals[0]:<10.2f} {mae_vals[1]:<10.2f} {mae_vals[2]:<10.2f} {mae_vals[3]:<10.2f}")
        
        # MAPE
        mape_vals = metrics_results['MAPE']
        print(f"{'':<10} {'MAPE':<8} {mape_vals[0]:<9.2f}% {mape_vals[1]:<9.2f}% {mape_vals[2]:<9.2f}% {mape_vals[3]:<9.2f}%")
        
        # RMSE
        rmse_vals = metrics_results['RMSE']
        print(f"{'':<10} {'RMSE':<8} {rmse_vals[0]:<10.2f} {rmse_vals[1]:<10.2f} {rmse_vals[2]:<10.2f} {rmse_vals[3]:<10.2f}")
        
        print("="*65 + "\n")
        
        # Save metrics to JSON for UI
        import json
        metrics_data = {
            "MAE": metrics_results['MAE'],
            "RMSE": metrics_results['RMSE'],
            "MAPE": metrics_results['MAPE']
        }
        with open('metrics.json', 'w') as f:
            json.dump(metrics_data, f)
        print(f"Metrics saved to metrics.json")

        # Return overall metrics
        return compute_metrics(all_preds, all_targets)
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MAE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history['val_mae'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(self.history['val_rmse'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Validation RMSE')
        axes[1, 0].grid(True)
        
        # MAPE
        axes[1, 1].plot(self.history['val_mape'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].set_title('Validation MAPE')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training history saved to {save_path}")
        plt.show()


# Main training script
if __name__ == "__main__":
    # Import your modules here
    from data_loader import PEMSBAYDataLoader
    from st_gnn import STGNN
    
    # Configuration
    DATA_DIR = r"E:\deeplearning project\dataset"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    SEQ_LEN = 12        # Input sequence length (1 hour)
    PRED_LEN = 12       # Prediction horizon (1 hour)
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    NUM_LAYERS = 3
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    print("="*50)
    print("TRAFFIC FLOW PREDICTION - ST-GNN")
    print("="*50)
    
    # Load data
    data_loader = PEMSBAYDataLoader(
        data_dir=DATA_DIR,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        train_ratio=0.7,
        val_ratio=0.1
    )
    
    # --- OPTIMIZATION FOR FASTER TRAINING ---
    print("\n[INFO] Subsetting data for faster training...")
    # Use only first 2000 samples for quick testing
    data_loader.train_data_scaled = data_loader.train_data_scaled[:2000]
    data_loader.val_data_scaled = data_loader.val_data_scaled[:500]
    data_loader.test_data_scaled = data_loader.test_data_scaled[:500]
    print(f"Subset sizes -> Train: {data_loader.train_data_scaled.shape}, Val: {data_loader.val_data_scaled.shape}")
    # ----------------------------------------
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Create model
    model = STGNN(
        num_nodes=data_loader.num_nodes,
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=1,
        num_layers=NUM_LAYERS,
        pred_len=PRED_LEN
    )
    
    # Create trainer
    trainer = Trainer(model, data_loader, DEVICE, LEARNING_RATE)
    
    # Train model
    trainer.train(
        epochs=EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        adj_mx=data_loader.adj_mx,
        save_path='best_st_gnn.pth'
    )
    
    # Plot training history
    trainer.plot_training_history('training_curves.png')
    
    # Test model
    trainer.test(test_loader, data_loader.adj_mx, 'best_st_gnn.pth')
    
    print("\n✓ All done!")
    # Force update timestamp