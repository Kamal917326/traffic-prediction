import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer
    Learns spatial dependencies between connected road sensors
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            output: (batch, num_nodes, out_features)
        """
        # Add self-loops to adjacency matrix
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = adj.sum(1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Graph convolution: AXW
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj_normalized, support)  # (batch, nodes, out_features)
        output = output + self.bias
        
        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network
    Captures temporal patterns using dilated causal convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvNet, self).__init__()
        
        # Causal padding: only look at past, not future
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        
        Returns:
            output: (batch, channels, seq_len)
        """
        # Apply convolution
        x = self.conv(x)
        
        # Remove future time steps (causal)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class SpatioTemporalBlock(nn.Module):
    """
    Combined Spatio-Temporal Block
    Applies graph convolution (spatial) followed by temporal convolution
    """
    def __init__(self, in_channels, out_channels, num_nodes, temporal_kernel=3, dilation=1):
        super(SpatioTemporalBlock, self).__init__()
        
        # Spatial convolution
        self.graph_conv = GraphConvolution(in_channels, out_channels)
        
        # Temporal convolution
        self.temporal_conv = TemporalConvNet(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel,
            dilation=dilation
        )
        
        # Residual connection (if dimensions match)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, seq_len, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)
        
        Returns:
            output: (batch, seq_len, num_nodes, out_channels)
        """
        batch_size, seq_len, num_nodes, in_channels = x.size()
        
        # Residual connection
        residual = x if self.residual is None else self.residual(x)
        
        # Apply graph convolution to each time step
        # Reshape: (batch * seq_len, num_nodes, in_channels)
        x_reshaped = x.reshape(batch_size * seq_len, num_nodes, in_channels)
        
        # Spatial convolution
        x_spatial = self.graph_conv(x_reshaped, adj)
        
        # Reshape back: (batch, seq_len, num_nodes, out_channels)
        x_spatial = x_spatial.reshape(batch_size, seq_len, num_nodes, -1)
        
        # Temporal convolution
        # Reshape to (batch * num_nodes, out_channels, seq_len)
        x_temp_input = x_spatial.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, -1, seq_len)
        
        # Apply temporal convolution
        x_temporal = self.temporal_conv(x_temp_input)
        
        # Reshape back: (batch, seq_len, num_nodes, out_channels)
        x_temporal = x_temporal.reshape(batch_size, num_nodes, -1, seq_len).permute(0, 3, 1, 2)
        
        # Add residual and normalize
        output = self.layer_norm(x_temporal + residual)
        
        return output


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network
    Main model for traffic flow prediction
    """
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, out_channels=1, 
                 num_layers=3, pred_len=12):
        super(STGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Stacked Spatio-Temporal blocks with increasing dilation
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_nodes=num_nodes,
                temporal_kernel=3,
                dilation=2**i  # Exponentially increasing dilation
            )
            for i in range(num_layers)
        ])
        
        # Output projection (predicts multiple future steps)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, out_channels * pred_len)
        )
    
    def forward(self, x, adj):
        """
        Args:
            x: Input sequence (batch, seq_len, num_nodes)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            predictions: (batch, pred_len, num_nodes)
        """
        batch_size, seq_len, num_nodes = x.size()
        
        # Add feature dimension: (batch, seq_len, num_nodes, 1)
        x = x.unsqueeze(-1)
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, num_nodes, hidden_channels)
        
        # Apply stacked Spatio-Temporal blocks
        for st_block in self.st_blocks:
            x = st_block(x, adj)
        
        # Use only the last time step for prediction
        x_last = x[:, -1, :, :]  # (batch, num_nodes, hidden_channels)
        
        # Project to output
        predictions = self.output_proj(x_last)  # (batch, num_nodes, pred_len * out_channels)
        
        # Reshape to (batch, pred_len, num_nodes)
        predictions = predictions.view(batch_size, num_nodes, self.pred_len).permute(0, 2, 1)
        
        return predictions
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example: Initialize and test the model
if __name__ == "__main__":
    # Model parameters
    num_nodes = 325  # PEMS-BAY has 325 sensors
    seq_len = 12     # Input sequence length
    pred_len = 12    # Prediction horizon
    batch_size = 32
    
    # Create model
    model = STGNN(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=64,
        out_channels=1,
        num_layers=3,
        pred_len=pred_len
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, num_nodes)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    
    # Forward pass
    predictions = model(x, adj)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Output shape: {predictions.shape}")
    print("\n✓ Model test successful!")