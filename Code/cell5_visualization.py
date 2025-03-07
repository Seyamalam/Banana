import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
import csv

from cell1_imports_and_constants import CLASS_NAMES, IDX_TO_CLASS, NUM_CLASSES

def save_figure(
    plt_figure, 
    base_filename, 
    formats=['png', 'svg'],
    dpi=300,
    close_after=True
):
    """
    Save a matplotlib figure in multiple formats.
    
    Args:
        plt_figure: Matplotlib figure or pyplot module
        base_filename: Base filename without extension
        formats: List of formats to save (e.g., ['png', 'svg'])
        dpi: Resolution for raster formats
        close_after: Whether to close the figure after saving
        
    Returns:
        List of saved file paths, one for each format
    """
    saved_paths = []
    
    for fmt in formats:
        filename = f"{base_filename}.{fmt}"
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the figure
        plt_figure.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_paths.append(filename)
        print(f"Figure saved to {filename}")
    
    if close_after:
        # Handle both Figure objects and pyplot module
        if plt_figure is plt:
            plt.close()  # Close the current figure window without clearing the current figure
        else:
            try:
                # For Figure objects
                plt.close(plt_figure)
            except:
                # Fallback if closing fails
                print(f"⚠️ Note: Could not close figure properly")
    
    # Return the file paths
    if len(saved_paths) == 2 and 'png' in formats and 'svg' in formats:
        png_path = next((path for path in saved_paths if path.endswith('.png')), None)
        svg_path = next((path for path in saved_paths if path.endswith('.svg')), None)
        return png_path, svg_path
    else:
        return saved_paths

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path=None, formats=None):
    """
    Plot training and validation metrics (loss and accuracy)
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        train_accs: list of training accuracies
        val_accs: list of validation accuracies
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, formats)
        
        # Also save metrics as CSV
        csv_path = f"{save_path}_metrics.csv"
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        })
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    plt.show()
    plt.close(fig)

def visualize_predictions(images, labels, preds, save_path=None, formats=None):
    """
    Visualize model predictions on sample images from the test set
    
    Args:
        images: batch of image tensors (B, C, H, W)
        labels: ground truth label tensors (B)
        preds: prediction tensors (B)
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    # Convert tensors to numpy arrays if they're not already
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    
    # Get number of samples
    num_samples = min(16, len(images))
    
    # Plot images with labels
    fig = plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        
        # Convert image from tensor format to display format
        img = np.transpose(images[i], (1, 2, 0))
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Set title color based on prediction correctness
        title_color = 'green' if preds[i] == labels[i] else 'red'
        true_label = IDX_TO_CLASS[labels[i]]
        pred_label = IDX_TO_CLASS[preds[i]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
    
    plt.tight_layout()
    
    if save_path:
        saved_paths = save_figure(fig, save_path, formats)
        
        # Save predictions as CSV
        csv_path = f"{save_path}_predictions.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sample_id', 'true_label', 'predicted_label', 'correct'])
            for i in range(num_samples):
                writer.writerow([
                    i,
                    IDX_TO_CLASS[labels[i]],
                    IDX_TO_CLASS[preds[i]],
                    preds[i] == labels[i]
                ])
        print(f"Predictions saved to {csv_path}")
        
        return saved_paths
    
    return fig

def visualize_sample_images(train_loader, save_path=None, formats=None):
    """
    Visualize sample images from the training set with their labels
    
    Args:
        train_loader: training data loader
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    # Get a batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Plot images with labels
    fig = plt.figure(figsize=(15, 8))
    for i in range(min(12, len(images))):
        ax = fig.add_subplot(3, 4, i+1, xticks=[], yticks=[])
        
        # Convert image from tensor format to display format
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"{IDX_TO_CLASS[labels[i]]}")
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, formats)
    
    plt.show()
    plt.close(fig)

def plot_confusion_matrix(cm, save_path=None, formats=None):
    """
    Plot confusion matrix
    
    Args:
        cm: confusion matrix
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    fig = plt.figure(figsize=(10, 8))
    
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    
    # Ensure confusion matrix is a numpy array
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Check if confusion matrix is valid
    if not isinstance(cm, np.ndarray) or cm.size == 0:
        print("Warning: Invalid confusion matrix")
        plt.close(fig)
        return
    
    # Normalize confusion matrix with error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1)
        cm_norm = np.zeros_like(cm, dtype=float)
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                cm_norm[i] = cm[i] / row_sum
    
    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, formats)
        
        # Save confusion matrix as CSV
        csv_path = f"{save_path}_matrix.csv"
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(csv_path)
        
        # Save normalized confusion matrix as CSV
        csv_norm_path = f"{save_path}_matrix_normalized.csv"
        cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        cm_norm_df.to_csv(csv_norm_path)
        
        print(f"Confusion matrices saved to {csv_path} and {csv_norm_path}")
    
    plt.show()
    plt.close(fig)

def plot_roc_curves(model, test_loader, device, save_path=None, formats=None):
    """
    Plot ROC curves for each class
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to run the model on
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store predictions and targets
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarize the labels for one-vs-rest ROC
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    fig = plt.figure(figsize=(12, 10))
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{IDX_TO_CLASS[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path, formats)
        
        # Save ROC data as CSV
        csv_path = f"{save_path}_roc_data.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'fpr', 'tpr', 'auc'])
            for i in range(NUM_CLASSES):
                writer.writerow([
                    IDX_TO_CLASS[i],
                    ','.join(map(str, fpr[i])),
                    ','.join(map(str, tpr[i])),
                    roc_auc[i]
                ])
        print(f"ROC data saved to {csv_path}")
    
    plt.show()
    plt.close(fig)

def plot_precision_recall_curves(model, test_loader, device, save_path=None, formats=None):
    """
    Plot precision-recall curves for each class
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to run the model on
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Store predictions and targets
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binarize the labels for one-vs-rest precision-recall
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    
    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        avg_precision[i] = np.mean(precision[i])
    
    # Plot precision-recall curves
    fig = plt.figure(figsize=(12, 10))
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{IDX_TO_CLASS[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_figure(fig, save_path, formats)
        
        # Save precision-recall data as CSV
        csv_path = f"{save_path}_pr_data.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'precision', 'recall', 'avg_precision'])
            for i in range(NUM_CLASSES):
                writer.writerow([
                    IDX_TO_CLASS[i],
                    ','.join(map(str, precision[i])),
                    ','.join(map(str, recall[i])),
                    avg_precision[i]
                ])
        print(f"Precision-recall data saved to {csv_path}")
    
    plt.show()
    plt.close(fig)

def visualize_model_architecture(model, input_size=(3, 224, 224), save_path=None, formats=None):
    """
    Visualize model architecture using a text-based summary
    
    Args:
        model: model to visualize
        input_size: input tensor size (channels, height, width)
        save_path: path to save the summary
        formats: not used, kept for API compatibility
    """
    # Create a string buffer to capture the model summary
    from io import StringIO
    import sys
    
    # Get model summary as string
    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer
    
    # Print model architecture
    print(model)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Restore stdout
    sys.stdout = original_stdout
    model_summary = buffer.getvalue()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as text file
        txt_path = f"{save_path}.txt"
        with open(txt_path, 'w') as f:
            f.write(model_summary)
        print(f"Model architecture saved to {txt_path}")
    
    # Print the summary
    print(model_summary)
    
    return model_summary

def plot_class_distribution(
    train_labels, 
    test_loader=None, 
    save_path=None, 
    formats=None
):
    """
    Plot class distribution in train and test sets
    
    Args:
        train_labels: List of labels in training set
        test_loader: DataLoader for test set (optional)
        save_path: Path to save the figure (without extension)
        formats: List of formats to save (default: ['png', 'svg'])
    
    Returns:
        Path to saved figure or None if save_path is None
    """
    if formats is None:
        formats = ['png', 'svg']
    
    # Count class frequencies in train set
    train_class_counts = {}
    for label in train_labels:
        if label not in train_class_counts:
            train_class_counts[label] = 0
        train_class_counts[label] += 1
    
    # Sort by class index
    train_class_counts = {k: train_class_counts.get(k, 0) for k in sorted(train_class_counts.keys())}
    
    # Get class names
    class_names = [IDX_TO_CLASS.get(i, f"Class {i}") for i in train_class_counts.keys()]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    
    # Plot train set distribution
    plt.bar(
        [str(name) for name in class_names], 
        list(train_class_counts.values()),
        label='Training Set'
    )
    
    # Plot test set distribution if provided
    if test_loader is not None:
        # Count class frequencies in test set
        test_class_counts = {}
        for _, labels in test_loader:
            for label in labels:
                label_idx = label.item()
                if label_idx not in test_class_counts:
                    test_class_counts[label_idx] = 0
                test_class_counts[label_idx] += 1
        
        # Sort by class index
        test_class_counts = {k: test_class_counts.get(k, 0) for k in sorted(test_class_counts.keys())}
        
        # Plot test set distribution
        plt.bar(
            [str(name) for name in class_names], 
            list(test_class_counts.values()),
            alpha=0.7,
            label='Test Set'
        )
    
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    return save_figure(plt, save_path, formats)

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

def visualize_feature_maps(model, test_loader, device, layer_name='conv1', num_samples=1, save_path=None, formats=None):
    """
    Visualize feature maps of a specific layer
    
    Args:
        model: trained model
        test_loader: test data loader
        device: device to run the model on
        layer_name: name of the layer to visualize
        num_samples: number of samples to visualize
        save_path: path to save the figure (without extension)
        formats: list of formats to save (default: ['png', 'svg'])
    """
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Register hook to get feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Get the layer
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(layer_name))
            break
    
    # Forward pass
    with torch.no_grad():
        images_device = images.to(device)
        _ = model(images_device)
    
    # Get feature maps
    feature_maps = activation[layer_name]
    
    # Plot feature maps
    for sample_idx in range(num_samples):
        # Get feature maps for this sample
        sample_maps = feature_maps[sample_idx]
        
        # Determine grid size
        num_features = sample_maps.size(0)
        grid_size = int(np.ceil(np.sqrt(num_features)))
        
        # Create figure
        fig = plt.figure(figsize=(15, 15))
        
        # Plot original image
        ax = fig.add_subplot(grid_size, grid_size, 1)
        img = images[sample_idx].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Original: {IDX_TO_CLASS[labels[sample_idx]]}")
        ax.axis('off')
        
        # Plot feature maps
        for i in range(min(num_features, grid_size*grid_size - 1)):
            ax = fig.add_subplot(grid_size, grid_size, i + 2)
            feature_map = sample_maps[i].cpu().numpy()
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"Filter {i+1}")
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            sample_save_path = f"{save_path}_sample{sample_idx+1}"
            save_figure(fig, sample_save_path, formats)
        
        plt.show()
        plt.close(fig) 