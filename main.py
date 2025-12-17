"""
Complete Traffic Flow Prediction Pipeline
Runs data loading, training, and visualization
"""

import torch
import os
import argparse

# Import your modules
from data_loader import PEMSBAYDataLoader
from st_gnn import STGNN
from train import Trainer
from visualize import create_prediction_report


def main(args):
    """
    Main pipeline for traffic flow prediction
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("TRAFFIC FLOW PREDICTION USING SPATIO-TEMPORAL GNN")
    print("="*70)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # =====================================================================
    # STEP 1: Load and Preprocess Data
    # =====================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = PEMSBAYDataLoader(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        subset_ratio=args.subset_ratio
    )
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size,
        shuffle=True
    )
    
    print(f"\n✓ Data loaded successfully")
    print(f"   Nodes (sensors): {data_loader.num_nodes}")
    print(f"   Total timesteps: {data_loader.num_timesteps}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # =====================================================================
    # STEP 2: Create Model
    # =====================================================================
    print("\n" + "="*70)
    print("STEP 2: CREATING MODEL")
    print("="*70)
    
    model = STGNN(
        num_nodes=data_loader.num_nodes,
        in_channels=1,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=args.num_layers,
        pred_len=args.pred_len
    )
    
    print(f"\n✓ Model created")
    print(f"   Architecture: ST-GNN")
    print(f"   Hidden channels: {args.hidden_channels}")
    print(f"   Number of layers: {args.num_layers}")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Input: {args.seq_len} time steps")
    print(f"   Output: {args.pred_len} time steps")
    
    # =====================================================================
    # STEP 3: Train Model
    # =====================================================================
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 3: TRAINING MODEL")
        print("="*70)
        
        trainer = Trainer(
            model=model,
            data_loader=data_loader,
            device=device,
            learning_rate=args.learning_rate
        )
        
        trainer.train(
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            adj_mx=data_loader.adj_mx,
            save_path=args.model_save_path
        )
        
        # Plot training history
        trainer.plot_training_history(args.training_curves_path)
        
        # Test model
        print("\n" + "="*70)
        print("TESTING MODEL")
        print("="*70)
        
        test_loss, test_mae, test_rmse, test_mape = trainer.test(
            test_loader=test_loader,
            adj_mx=data_loader.adj_mx,
            model_path=args.model_save_path
        )
        
    else:
        print("\n⏭️  Skipping training (loading pre-trained model)...")
    
    # =====================================================================
    # STEP 4: Visualization and Analysis
    # =====================================================================
    if not args.skip_visualization:
        print("\n" + "="*70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Load best model
        checkpoint = torch.load(args.model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Create comprehensive report
        create_prediction_report(
            model=model,
            test_loader=test_loader,
            adj_mx=data_loader.adj_mx,
            data_loader=data_loader,
            device=device,
            pred_len=args.pred_len
        )
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE ✓")
    print("="*70)
    
    print("\n📂 Generated Files:")
    if not args.skip_training:
        print(f"   ✓ {args.model_save_path} - Trained model")
        print(f"   ✓ {args.training_curves_path} - Training curves")
    
    print("\n" + "="*70)
    print("All done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Flow Prediction with ST-GNN")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default=r"e:\deeplearning project\dataset",
                       help='Path to data directory')
    parser.add_argument('--seq_len', type=int, default=12,
                       help='Input sequence length (default: 12)')
    parser.add_argument('--pred_len', type=int, default=12,
                       help='Prediction horizon (default: 12)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio (default: 0.1)')
    parser.add_argument('--subset_ratio', type=float, default=0.5, 
                       help='Fraction of dataset to use (default: 0.5 = 50%)')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden dimension size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of ST blocks (default: 3)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Output paths
    parser.add_argument('--model_save_path', type=str, default='best_st_gnn.pth',
                       help='Path to save model (default: best_st_gnn.pth)')
    parser.add_argument('--training_curves_path', type=str, default='training_curves.png',
                       help='Path to save training curves (default: training_curves.png)')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load existing model')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Run main pipeline
    main(args)