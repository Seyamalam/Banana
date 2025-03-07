import os
import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import json
from typing import Dict, List, Tuple, Any, Optional

from cell1_imports_and_constants import IDX_TO_CLASS, NUM_CLASSES

def save_model(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, save_path):
    """
    Save model checkpoint with training information
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, optimizer, load_path, device):
    """
    Load model checkpoint with training information
    """
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        return model, optimizer, 0, [], [], [], []
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', [])
    val_loss = checkpoint.get('val_loss', [])
    train_acc = checkpoint.get('train_acc', [])
    val_acc = checkpoint.get('val_acc', [])
    
    print(f"Model loaded from {load_path} (epoch {epoch})")
    return model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc

def calculate_model_size(model):
    """
    Calculate model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def evaluate_model(model, test_loader, device):
    """
    Evaluate a model on a test dataset and compute metrics.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics, true labels, and predicted labels
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Ensure confusion matrix is a numpy array
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Calculate normalized confusion matrix
    if isinstance(cm, np.ndarray) and cm.size > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1)
            cm_norm = np.zeros_like(cm, dtype=float)
            for i, row_sum in enumerate(row_sums):
                if row_sum > 0:
                    cm_norm[i] = cm[i] / row_sum
    else:
        cm_norm = np.array([[0]])
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm
    }, true_labels, predictions

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        
        # Also save as CSV
        csv_path = f"{os.path.splitext(save_path)[0]}.csv"
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(csv_path)
        print(f"Confusion matrix data saved to {csv_path}")
        
        # Save normalized confusion matrix
        csv_norm_path = f"{os.path.splitext(save_path)[0]}_normalized.csv"
        cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        cm_norm_df.to_csv(csv_norm_path)
        print(f"Normalized confusion matrix data saved to {csv_norm_path}")
    
    plt.show()

def save_classification_report(report, save_path):
    """
    Save classification report as CSV
    
    Args:
        report: classification report string from sklearn
        save_path: path to save the report
    """
    # Parse the report
    lines = report.split('\n')
    classes = []
    data = []
    
    for line in lines[2:-5]:  # Skip header and footer
        if line.strip():
            row_data = line.strip().split()
            if len(row_data) >= 5:  # Valid data row
                class_name = row_data[0]
                precision = float(row_data[1])
                recall = float(row_data[2])
                f1_score = float(row_data[3])
                support = int(row_data[4])
                
                classes.append(class_name)
                data.append([class_name, precision, recall, f1_score, support])
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data, columns=['class', 'precision', 'recall', 'f1_score', 'support'])
    df.to_csv(save_path, index=False)
    print(f"Classification report saved to {save_path}")

def export_model_summary(model, save_path):
    """
    Export model summary information to JSON
    
    Args:
        model: PyTorch model
        save_path: path to save the summary
    """
    # Get model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = calculate_model_size(model)
    
    # Create layers information
    layers_info = []
    for name, module in model.named_children():
        layer_info = {
            'name': name,
            'type': module.__class__.__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
        }
        
        # Add specific information based on layer type
        if isinstance(module, torch.nn.Conv2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                'stride': module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                'padding': module.padding[0] if isinstance(module.padding, tuple) else module.padding,
            })
        elif isinstance(module, torch.nn.Linear):
            layer_info.update({
                'in_features': module.in_features,
                'out_features': module.out_features,
            })
        
        layers_info.append(layer_info)
    
    # Create summary dictionary
    summary = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'layers': layers_info
    }
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Model summary saved to {save_path}")
    
    # Also save as CSV for easier viewing
    csv_path = f"{os.path.splitext(save_path)[0]}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['layer_name', 'layer_type', 'parameters', 'details'])
        
        for layer in layers_info:
            details = []
            for k, v in layer.items():
                if k not in ['name', 'type', 'parameters']:
                    details.append(f"{k}={v}")
            
            writer.writerow([
                layer['name'],
                layer['type'],
                layer['parameters'],
                ', '.join(details)
            ])
        
        # Add summary row
        writer.writerow(['', '', '', ''])
        writer.writerow(['Total Parameters', '', total_params, ''])
        writer.writerow(['Trainable Parameters', '', trainable_params, ''])
        writer.writerow(['Model Size (MB)', '', model_size_mb, ''])
    
    print(f"Model summary also saved as CSV to {csv_path}")

def save_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Save training history as CSV
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        train_accs: list of training accuracies
        val_accs: list of validation accuracies
        save_path: path to save the history
    """
    # Create DataFrame
    history = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    
    # Save as CSV
    history.to_csv(save_path, index=False)
    print(f"Training history saved to {save_path}")

def calculate_per_class_metrics(all_targets, all_preds, save_path=None):
    """
    Calculate and optionally save per-class metrics
    
    Args:
        all_targets: list of true labels
        all_preds: list of predicted labels
        save_path: path to save the metrics (optional)
    
    Returns:
        DataFrame with per-class metrics
    """
    class_metrics = []
    
    for i in range(NUM_CLASSES):
        # True positives, false positives, false negatives
        tp = sum(1 for j in range(len(all_targets)) if all_targets[j] == i and all_preds[j] == i)
        fp = sum(1 for j in range(len(all_targets)) if all_targets[j] != i and all_preds[j] == i)
        fn = sum(1 for j in range(len(all_targets)) if all_targets[j] == i and all_preds[j] != i)
        tn = sum(1 for j in range(len(all_targets)) if all_targets[j] != i and all_preds[j] != i)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        class_metrics.append({
            'class': IDX_TO_CLASS[i],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': tp + fn
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(class_metrics)
    
    # Save if path provided
    if save_path:
        metrics_df.to_csv(save_path, index=False)
        print(f"Per-class metrics saved to {save_path}")
    
    return metrics_df

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer], 
                   epoch: int, 
                   val_acc: float, 
                   save_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        val_acc: Validation accuracy
        save_path: Path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'val_acc': val_acc
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer], 
                   checkpoint_path: str) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Tuple of (model, optimizer, epoch, val_acc)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and validation accuracy
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    return model, optimizer, epoch, val_acc

def get_model_summary(model: torch.nn.Module) -> str:
    """
    Get a string summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        String summary of the model
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create summary string
    summary = []
    summary.append(str(model))
    summary.append(f"\nTotal parameters: {total_params:,}")
    summary.append(f"Trainable parameters: {trainable_params:,}")
    
    return "\n".join(summary)

def calculate_class_weights(train_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        train_loader: Training data loader
        
    Returns:
        Tensor of class weights
    """
    # Count classes
    class_counts = {}
    for _, labels in train_loader:
        for label in labels:
            label_item = label.item()
            if label_item in class_counts:
                class_counts[label_item] += 1
            else:
                class_counts[label_item] = 1
    
    # Get number of classes
    num_classes = len(class_counts)
    
    # Calculate weights
    total_samples = sum(class_counts.values())
    weights = torch.zeros(num_classes)
    
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights

def export_model_info(model: torch.nn.Module, 
                     metrics: Dict[str, Any], 
                     export_path: str) -> None:
    """
    Export model information and metrics to a text file.
    
    Args:
        model: PyTorch model
        metrics: Dictionary of evaluation metrics
        export_path: Path to save the information
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Get model summary
    model_summary = get_model_summary(model)
    
    # Format metrics
    metrics_str = []
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            metrics_str.append(f"{key}: {value:.4f}")
    
    # Combine information
    info = []
    info.append("MODEL INFORMATION")
    info.append("=" * 50)
    info.append(model_summary)
    info.append("\nEVALUATION METRICS")
    info.append("=" * 50)
    info.extend(metrics_str)
    
    # Write to file
    with open(export_path, 'w') as f:
        f.write("\n".join(info))
    
    print(f"Model information exported to {export_path}")

if __name__ == "__main__":
    print("Utility functions loaded successfully.")
    print("Use save_checkpoint() to save model checkpoints.")
    print("Use load_checkpoint() to load model checkpoints.")
    print("Use evaluate_model() to evaluate model performance.") 