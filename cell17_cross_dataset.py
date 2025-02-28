import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Tuple, Optional, Union, Any
from cell5_visualization import save_figure
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ExternalDataset(Dataset):
    """Dataset class for loading external image datasets."""
    
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (128, 128),
        class_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset
            transform: Transforms to apply to images
            target_size: Target image size
            class_map: Mapping from class names to indices
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.class_map = class_map
        
        # Find all image files
        self.image_paths = []
        self.labels = []
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Scan directory structure
        if os.path.exists(root_dir):
            # Check if dataset has class subdirectories
            subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
            if subdirs:
                # Dataset has class subdirectories
                for class_name in subdirs:
                    class_dir = os.path.join(root_dir, class_name)
                    class_idx = self.class_map.get(class_name, -1) if self.class_map else -1
                    
                    if class_idx >= 0:  # Only include classes in the class map
                        for ext in ['*.jpg', '*.jpeg', '*.png']:
                            image_paths = glob.glob(os.path.join(class_dir, ext))
                            self.image_paths.extend(image_paths)
                            self.labels.extend([class_idx] * len(image_paths))
            else:
                # Flat directory structure, assume all images are of the same class
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_paths = glob.glob(os.path.join(root_dir, ext))
                    self.image_paths.extend(image_paths)
                    self.labels.extend([0] * len(image_paths))  # Assign class 0 to all images
        
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, label


def evaluate_cross_dataset(
    model: nn.Module,
    model_name: str,
    datasets: List[Dict[str, Any]],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Evaluate model performance across multiple datasets.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        datasets: List of dictionaries, each containing:
                 - 'name': Dataset name
                 - 'loader': DataLoader for the dataset, or
                 - 'root_dir': Root directory of the dataset
                 - 'class_map': Mapping from class names to indices
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        
    Returns:
        DataFrame with evaluation results, path to CSV file, and path to plot
    """
    # Set model to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Store results
    results = []
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        print(f"Evaluating on {dataset_name}...")
        
        # Get data loader
        if 'loader' in dataset_info:
            loader = dataset_info['loader']
        else:
            # Create dataset and loader
            dataset = ExternalDataset(
                root_dir=dataset_info['root_dir'],
                class_map=dataset_info['class_map']
            )
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4
            )
        
        # Evaluate
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': all_targets,
            'y_pred': all_preds
        })
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'dataset': r['dataset'],
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1']
        }
        for r in results
    ])
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_cross_dataset_evaluation.csv")
    df.to_csv(csv_path, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    x = np.arange(len(df))
    width = 0.2
    
    plt.bar(x - width*1.5, df['accuracy'], width, label='Accuracy')
    plt.bar(x - width/2, df['precision'], width, label='Precision')
    plt.bar(x + width/2, df['recall'], width, label='Recall')
    plt.bar(x + width*1.5, df['f1'], width, label='F1')
    
    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title(f'Cross-Dataset Evaluation - {model_name}')
    plt.xticks(x, df['dataset'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, f"{model_name}_cross_dataset_evaluation")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, png_path


def compare_cross_dataset_performance(
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Compare cross-dataset performance of multiple models.
    
    Args:
        models: List of dictionaries, each containing:
               - 'name': Model name
               - 'model': PyTorch model
        datasets: List of dictionaries, each containing:
                 - 'name': Dataset name
                 - 'loader': DataLoader for the dataset, or
                 - 'root_dir': Root directory of the dataset
                 - 'class_map': Mapping from class names to indices
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results, path to CSV file, and path to plot
    """
    # Store results
    all_results = []
    
    for model_dict in models:
        model_name = model_dict['name']
        model = model_dict['model']
        
        print(f"Evaluating {model_name} across datasets...")
        
        # Evaluate on each dataset
        for dataset_info in datasets:
            dataset_name = dataset_info['name']
            
            # Get data loader
            if 'loader' in dataset_info:
                loader = dataset_info['loader']
            else:
                # Create dataset and loader
                dataset = ExternalDataset(
                    root_dir=dataset_info['root_dir'],
                    class_map=dataset_info['class_map']
                )
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=4
                )
            
            # Evaluate
            model = model.to(device)
            model.eval()
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )
            
            # Store results
            all_results.append({
                'model': model_name,
                'dataset': dataset_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cross_dataset_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Create heatmap visualization
    plt.figure(figsize=(12, 8))
    
    # Pivot table for heatmap
    pivot_df = df.pivot(index='model', columns='dataset', values='accuracy')
    
    # Plot heatmap
    plt.imshow(pivot_df.values, cmap='YlGnBu')
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            plt.text(j, i, f'{pivot_df.values[i, j]:.3f}', ha='center', va='center')
    
    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Model')
    plt.title('Cross-Dataset Accuracy Comparison')
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45, ha='right')
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.colorbar(label='Accuracy')
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "cross_dataset_comparison_heatmap")
    heatmap_png, heatmap_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # Create bar chart visualization
    plt.figure(figsize=(14, 8))
    
    # Group by model and calculate mean accuracy across datasets
    model_avg = df.groupby('model')['accuracy'].mean().reset_index()
    model_avg = model_avg.sort_values('accuracy', ascending=False)
    
    # Plot bar chart
    plt.bar(model_avg['model'], model_avg['accuracy'])
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy Across Datasets')
    plt.title('Model Generalization Performance')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(model_avg['accuracy']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "cross_dataset_comparison_bar")
    bar_png, bar_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, heatmap_png


def calculate_generalization_gap(
    models: List[Dict[str, Any]],
    train_loader: torch.utils.data.DataLoader,
    test_loaders: List[Dict[str, Any]],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Calculate generalization gap between training data and external datasets.
    
    Args:
        models: List of dictionaries, each containing:
               - 'name': Model name
               - 'model': PyTorch model
        train_loader: DataLoader for training data
        test_loaders: List of dictionaries, each containing:
                     - 'name': Dataset name
                     - 'loader': DataLoader for the dataset
        device: Device to run evaluation on
        output_dir: Directory to save results
        
    Returns:
        DataFrame with generalization gaps, path to CSV file, and path to plot
    """
    # Store results
    results = []
    
    for model_dict in models:
        model_name = model_dict['name']
        model = model_dict['model']
        
        print(f"Calculating generalization gap for {model_name}...")
        
        # Evaluate on training data
        model = model.to(device)
        model.eval()
        
        train_preds = []
        train_targets = []
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
        
        train_accuracy = accuracy_score(train_targets, train_preds)
        
        # Evaluate on each test dataset
        for test_info in test_loaders:
            test_name = test_info['name']
            test_loader = test_info['loader']
            
            test_preds = []
            test_targets = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    test_preds.extend(preds.cpu().numpy())
                    test_targets.extend(targets.cpu().numpy())
            
            test_accuracy = accuracy_score(test_targets, test_preds)
            
            # Calculate generalization gap
            gap = train_accuracy - test_accuracy
            
            # Store results
            results.append({
                'model': model_name,
                'train_dataset': 'Training Data',
                'test_dataset': test_name,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'generalization_gap': gap,
                'relative_gap': gap / train_accuracy if train_accuracy > 0 else 0
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "generalization_gap_analysis.csv")
    df.to_csv(csv_path, index=False)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Group by model and calculate mean generalization gap
    model_avg = df.groupby('model')['relative_gap'].mean().reset_index()
    model_avg = model_avg.sort_values('relative_gap')
    
    # Plot bar chart
    plt.bar(model_avg['model'], model_avg['relative_gap'] * 100)  # Convert to percentage
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average Generalization Gap (%)')
    plt.title('Model Generalization Gap Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(model_avg['relative_gap']):
        plt.text(i, v * 100 + 1, f'{v * 100:.1f}%', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "generalization_gap_analysis")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, png_path


if __name__ == "__main__":
    # Example usage
    print("Cross-dataset evaluation module loaded successfully.")
    print("Use evaluate_cross_dataset() to test model performance on external datasets.")
    print("Use compare_cross_dataset_performance() to compare models across datasets.")
    print("Use calculate_generalization_gap() to analyze generalization capabilities.") 