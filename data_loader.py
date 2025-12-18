import numpy as np
import pandas as pd
import h5py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class PEMSBAYDataLoader:
    """
    Loads and preprocesses PEMS-BAY traffic dataset
    """
    def __init__(self, data_dir, seq_len=12, pred_len=12, train_ratio=0.7, val_ratio=0.1, subset_ratio=1.0):
        """
        Args:
            data_dir: Path to folder containing pems-bay.h5 and adj_mx_bay.pkl
            seq_len: Input sequence length (number of time steps to look back)
            pred_len: Prediction length (how many steps ahead to predict)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.subset_ratio = subset_ratio
        
        self.subset_ratio = subset_ratio
        
        # Check and unzip data if needed
        h5_path = os.path.join(self.data_dir, 'pems-bay.h5')
        zip_path = os.path.join(self.data_dir, 'pems-bay.zip')
        
        if not os.path.exists(h5_path) and os.path.exists(zip_path):
            print(f"Dataset not found at {h5_path}. Attempting to unzip from {zip_path}...")
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("Unzip successful!")
            except Exception as e:
                print(f"Error unzipping dataset: {e}")
        
        # Load data
        print("Loading PEMS-BAY dataset...")
        self.load_data()
        self.load_adjacency_matrix()
        
        # Preprocess
        print("Preprocessing data...")
        self.preprocess_data()
        
        print(f"\nDataset loaded successfully!")
        print(f"Number of sensors: {self.num_nodes}")
        print(f"Total time steps: {self.num_timesteps}")
        print(f"Train size: {self.train_data_scaled.shape}")
        print(f"Val size: {self.val_data_scaled.shape}")
        print(f"Test size: {self.test_data_scaled.shape}")
    
    def load_data(self):
        """Load traffic speed data from H5 file"""
        h5_file = os.path.join(self.data_dir, 'pems-bay.h5')
        
        with h5py.File(h5_file, 'r') as f:
            # Print available keys to understand structure
            print(f"Available keys in H5 file: {list(f.keys())}")
            
            # Usually the data is in 'speed' or 'data' key
            if 'speed' in f.keys():
                data = f['speed']
                if isinstance(data, h5py.Group):
                    self.raw_data = data['block0_values'][:]
                else:
                    self.raw_data = data[:]
            elif 'data' in f.keys():
                data = f['data']
                if isinstance(data, h5py.Group):
                    self.raw_data = data['block0_values'][:]
                else:
                    self.raw_data = data[:]
            else:
                # Take the first key
                key = list(f.keys())[0]
                data = f[key]
                if isinstance(data, h5py.Group) and 'block0_values' in data.keys():
                    self.raw_data = data['block0_values'][:]
                else:
                    self.raw_data = data[:]
        
        print(f"Raw data shape: {self.raw_data.shape}")
        # Expected shape: (num_timesteps, num_sensors) or (num_sensors, num_timesteps)
        
        # Ensure shape is (num_timesteps, num_sensors)
        if self.raw_data.shape[0] < self.raw_data.shape[1]:
            self.raw_data = self.raw_data.T
        
        self.num_timesteps, self.num_nodes = self.raw_data.shape
        print(f"Reshaped to: timesteps={self.num_timesteps}, sensors={self.num_nodes}")
    
    def load_adjacency_matrix(self):
        """Load graph adjacency matrix from PKL file"""
        pkl_file = os.path.join(self.data_dir, 'adj_mx_bay.pkl')
        
        with open(pkl_file, 'rb') as f:
            try:
                # Try different pickle protocols
                adj_mx = pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                adj_mx = pickle.load(f)
        
        # Handle different adjacency matrix formats
        if isinstance(adj_mx, (tuple, list)):
            # Sometimes stored as (ids, ids, adj_matrix)
            self.adj_mx = adj_mx[2] if len(adj_mx) > 2 else adj_mx[0]
        else:
            self.adj_mx = adj_mx
        
        print(f"Adjacency matrix shape: {self.adj_mx.shape}")
        print(f"Adjacency matrix sparsity: {(self.adj_mx == 0).sum() / self.adj_mx.size * 100:.2f}%")
        
        # Convert to torch tensor
        self.adj_mx = torch.FloatTensor(self.adj_mx)
    
    def preprocess_data(self):
        """Normalize data and create train/val/test splits"""
        # Handle missing values (if any)
        self.raw_data = np.nan_to_num(self.raw_data, nan=0.0)
        
        # Apply subset ratio
        if self.subset_ratio < 1.0:
            original_len = len(self.raw_data)
            subset_len = int(original_len * self.subset_ratio)
            self.raw_data = self.raw_data[:subset_len]
            self.num_timesteps = subset_len
            print(f"Dataset subsetting: {original_len} -> {subset_len} timesteps ({self.subset_ratio*100:.1f}%)")
        
        # Split data
        train_size = int(self.num_timesteps * self.train_ratio)
        val_size = int(self.num_timesteps * self.val_ratio)
        
        train_data = self.raw_data[:train_size]
        val_data = self.raw_data[train_size:train_size + val_size]
        test_data = self.raw_data[train_size + val_size:]
        
        # Fit scaler on training data only
        self.scaler = StandardScaler()
        train_data_scaled = self.scaler.fit_transform(train_data)
        val_data_scaled = self.scaler.transform(val_data)
        test_data_scaled = self.scaler.transform(test_data)
        
        # Store scaled data directly instead of sequences
        self.train_data_scaled = train_data_scaled
        self.val_data_scaled = val_data_scaled
        self.test_data_scaled = test_data_scaled
    
    def _create_sequences(self, data):
        """Deprecated: uses too much memory"""
        pass
    
    def get_dataloaders(self, batch_size=64, shuffle=True):
        """
        Create PyTorch DataLoaders for train/val/test
        
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset = TrafficDataset(self.train_data_scaled, self.seq_len, self.pred_len)
        val_dataset = TrafficDataset(self.val_data_scaled, self.seq_len, self.pred_len)
        test_dataset = TrafficDataset(self.test_data_scaled, self.seq_len, self.pred_len)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data):
        """Convert normalized data back to original scale"""
        return self.scaler.inverse_transform(data)


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic data - Memory Efficient"""
    
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data) - seq_len - pred_len + 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# Example usage and data exploration
if __name__ == "__main__":
    # Path to your data folder
    data_dir = r"E:\deeplearning project\dataset"
    
    # Initialize data loader
    loader = PEMSBAYDataLoader(
        data_dir=data_dir,
        seq_len=12,      # Look back 12 time steps (1 hour if 5-min intervals)
        pred_len=12,     # Predict 12 steps ahead (1 hour)
        train_ratio=0.7,
        val_ratio=0.1
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=32)
    
    # Test: Get one batch
    print("\n--- Testing Data Loading ---")
    for batch_x, batch_y in train_loader:
        print(f"Batch input shape: {batch_x.shape}")   # (batch, seq_len, num_nodes)
        print(f"Batch target shape: {batch_y.shape}")  # (batch, pred_len, num_nodes)
        print(f"Adjacency matrix shape: {loader.adj_mx.shape}")
        break
    
    print("\n✓ Data loading successful!")