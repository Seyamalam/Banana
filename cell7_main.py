import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from cell1_imports_and_constants import set_seed, NUM_EPOCHS, NUM_CLASSES
from cell2_dataset import load_data
from cell3_model import BananaLeafCNN
from cell4_training import train, validate
from cell5_visualization import (
    plot_training_metrics, visualize_predictions, visualize_sample_images,
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    visualize_model_architecture, plot_class_distribution, save_classification_report,
    visualize_feature_maps
)
from cell6_utils import (
    save_model, load_model, calculate_model_size, evaluate_model, 
    export_model_summary, save_training_history, calculate_per_class_metrics
)

def parse_args():
    parser = argparse.ArgumentParser(description='Banana Leaf Disease Classification')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to save models')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--save_formats', type=str, default='png,svg', help='Formats to save visualizations (comma-separated)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse save formats
    save_formats = args.save_formats.split(',')
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    results_dir = os.path.join(args.model_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    vis_dir = os.path.join(results_dir, 'visualizations')
    metrics_dir = os.path.join(results_dir, 'metrics')
    model_info_dir = os.path.join(results_dir, 'model_info')
    
    for directory in [vis_dir, metrics_dir, model_info_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_data(args.data_dir, batch_size=args.batch_size)
    
    # Visualize class distribution
    plot_class_distribution(
        train_loader, test_loader, 
        save_path=os.path.join(vis_dir, 'class_distribution'),
        formats=save_formats
    )
    
    # Visualize sample images
    if not args.eval_only:
        visualize_sample_images(
            train_loader, 
            save_path=os.path.join(vis_dir, 'sample_images'),
            formats=save_formats
        )
    
    # Create model
    model = BananaLeafCNN(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Export model architecture visualization
    visualize_model_architecture(
        model, 
        save_path=os.path.join(model_info_dir, 'model_architecture'),
        formats=save_formats
    )
    
    # Export model summary
    export_model_summary(
        model,
        save_path=os.path.join(model_info_dir, 'model_summary.json')
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if resuming
    checkpoint_path = os.path.join(args.model_dir, 'best_model.pth')
    start_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    
    if args.resume or args.eval_only:
        model, optimizer, start_epoch, train_losses, val_losses, train_accs, val_accs = load_model(
            model, optimizer, checkpoint_path, device
        )
        if len(val_accs) > 0:
            best_val_acc = max(val_accs)
            
        # Save training history if available
        if len(train_losses) > 0:
            save_training_history(
                train_losses, val_losses, train_accs, val_accs,
                save_path=os.path.join(metrics_dir, 'training_history.csv')
            )
    
    # Evaluation only mode
    if args.eval_only:
        print("Running evaluation only...")
        
        # Create evaluation directory
        eval_dir = os.path.join(results_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Run evaluation with saving metrics
        metrics = evaluate_model(model, test_loader, criterion, device, save_dir=eval_dir)
        
        print(f"Test Loss: {metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
        print(f"Total inference time: {metrics['inference_time']:.4f} seconds")
        print(f"Average inference time per sample: {metrics['avg_inference_time']*1000:.2f} ms")
        
        # Calculate and save per-class metrics
        calculate_per_class_metrics(
            metrics['all_targets'], metrics['all_preds'],
            save_path=os.path.join(eval_dir, 'per_class_metrics.csv')
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            save_path=os.path.join(vis_dir, 'confusion_matrix'),
            formats=save_formats
        )
        
        # Plot ROC curves
        plot_roc_curves(
            model, test_loader, device,
            save_path=os.path.join(vis_dir, 'roc_curves'),
            formats=save_formats
        )
        
        # Plot precision-recall curves
        plot_precision_recall_curves(
            model, test_loader, device,
            save_path=os.path.join(vis_dir, 'precision_recall_curves'),
            formats=save_formats
        )
        
        # Visualize predictions
        visualize_predictions(
            model, test_loader, device, 
            save_path=os.path.join(vis_dir, 'predictions'),
            formats=save_formats
        )
        
        # Visualize feature maps for first convolutional layer
        visualize_feature_maps(
            model, test_loader, device, layer_name='conv1',
            save_path=os.path.join(vis_dir, 'feature_maps_conv1'),
            formats=save_formats
        )
        
        return
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, report = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save classification report
        epoch_report_path = os.path.join(metrics_dir, f'classification_report_epoch{epoch+1}.csv')
        save_classification_report(report, epoch_report_path)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Classification Report:\n{report}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model, optimizer, epoch+1, 
                train_losses, val_losses, train_accs, val_accs,
                checkpoint_path
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Save training history at best epoch
            save_training_history(
                train_losses, val_losses, train_accs, val_accs,
                save_path=os.path.join(metrics_dir, 'best_training_history.csv')
            )
        
        # Save latest model
        save_model(
            model, optimizer, epoch+1, 
            train_losses, val_losses, train_accs, val_accs,
            os.path.join(args.model_dir, 'latest_model.pth')
        )
        
        # Save training history every epoch
        save_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path=os.path.join(metrics_dir, 'training_history.csv')
        )
    
    # Plot training metrics
    plot_training_metrics(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(vis_dir, 'training_metrics'),
        formats=save_formats
    )
    
    # Final evaluation
    print("Running final evaluation...")
    
    # Create evaluation directory
    final_eval_dir = os.path.join(results_dir, 'final_evaluation')
    os.makedirs(final_eval_dir, exist_ok=True)
    
    # Run evaluation with saving metrics
    metrics = evaluate_model(model, test_loader, criterion, device, save_dir=final_eval_dir)
    
    print(f"Final Test Loss: {metrics['test_loss']:.4f}")
    print(f"Final Test Accuracy: {metrics['test_acc']:.2f}%")
    
    # Calculate and save per-class metrics
    calculate_per_class_metrics(
        metrics['all_targets'], metrics['all_preds'],
        save_path=os.path.join(final_eval_dir, 'per_class_metrics.csv')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        save_path=os.path.join(vis_dir, 'final_confusion_matrix'),
        formats=save_formats
    )
    
    # Plot ROC curves
    plot_roc_curves(
        model, test_loader, device,
        save_path=os.path.join(vis_dir, 'final_roc_curves'),
        formats=save_formats
    )
    
    # Plot precision-recall curves
    plot_precision_recall_curves(
        model, test_loader, device,
        save_path=os.path.join(vis_dir, 'final_precision_recall_curves'),
        formats=save_formats
    )
    
    # Visualize predictions
    visualize_predictions(
        model, test_loader, device, 
        save_path=os.path.join(vis_dir, 'final_predictions'),
        formats=save_formats
    )
    
    # Visualize feature maps for first convolutional layer
    visualize_feature_maps(
        model, test_loader, device, layer_name='conv1',
        save_path=os.path.join(vis_dir, 'final_feature_maps_conv1'),
        formats=save_formats
    )

if __name__ == "__main__":
    main() 