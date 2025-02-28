import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from cell5_visualization import save_figure
from cell4_training import train, validate
from cell11_training_resources import measure_training_resources


class AblationStudy:
    """Class to manage ablation studies for neural network models."""
    
    def __init__(
        self, 
        base_model: nn.Module,
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer_fn: Callable = torch.optim.Adam,
        lr: float = 0.001,
        epochs: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'models'
    ):
        """
        Initialize the ablation study.
        
        Args:
            base_model: Base model to perform ablation on
            model_name: Name of the base model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer_fn: Function to create optimizer
            lr: Learning rate
            epochs: Number of epochs for training
            device: Device to run on ('cuda' or 'cpu')
            output_dir: Directory to save results
        """
        self.base_model = base_model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results
        self.results = []
        
    def _train_and_evaluate(self, model: nn.Module, variant_name: str) -> Dict[str, Any]:
        """
        Train and evaluate a model variant
        
        Args:
            model: Model variant to train and evaluate
            variant_name: Name of the variant
            
        Returns:
            Dictionary of training results
        """
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer
        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Initialize start_time to measure training duration
        start_time = time.time()
        
        for epoch in range(self.epochs):
            train_loss, train_acc = train(
                model, self.train_loader, self.criterion, optimizer, self.device
            )
            val_loss, val_acc, val_report = validate(
                model, self.val_loader, self.criterion, self.device
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Variant: {variant_name}, Epoch: {epoch+1}/{self.epochs}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        training_time = time.time() - start_time
        
        # Get final predictions for confusion matrix
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Measure inference time
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.to(self.device)
                _ = model(inputs)
        inference_time = (time.time() - start_time) / len(self.val_loader.dataset)
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{variant_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Calculate model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return {
            'variant_name': variant_name,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'training_time': training_time,
            'inference_time': inference_time,
            'num_params': num_params,
            'model_size_mb': model_size_mb,
            'y_true': all_targets,
            'y_pred': all_preds
        }
    
    def run_ablation_studies(self, variants: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Run ablation studies on model variants.
        
        Args:
            variants: List of dictionaries, each containing:
                     - 'name': Name of the variant
                     - 'model': Modified model for this variant
                     - 'description': Description of the modification
                     
        Returns:
            DataFrame with ablation study results
        """
        # First, evaluate the base model
        print(f"Evaluating base model: {self.model_name}")
        base_results = self._train_and_evaluate(self.base_model, self.model_name)
        self.results.append(base_results)
        
        # Then evaluate each variant
        for variant in variants:
            print(f"Evaluating variant: {variant['name']}")
            variant_results = self._train_and_evaluate(variant['model'], variant['name'])
            variant_results['description'] = variant['description']
            self.results.append(variant_results)
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'variant': r['variant_name'],
                'description': r.get('description', 'Base model'),
                'val_accuracy': r['final_val_acc'],
                'train_accuracy': r['final_train_acc'],
                'val_loss': r['final_val_loss'],
                'train_loss': r['final_train_loss'],
                'parameters': r['num_params'],
                'model_size_mb': r['model_size_mb'],
                'training_time_s': r['training_time'],
                'inference_time_ms': r['inference_time'] * 1000
            }
            for r in self.results
        ])
        
        # Save results to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_ablation_study.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        self._create_visualizations()
        
        return df
    
    def _create_visualizations(self) -> None:
        """Create visualizations for ablation study results."""
        # Extract data for plotting
        variants = [r['variant_name'] for r in self.results]
        val_accs = [r['final_val_acc'] for r in self.results]
        train_accs = [r['final_train_acc'] for r in self.results]
        params = [r['num_params'] for r in self.results]
        inference_times = [r['inference_time'] * 1000 for r in self.results]  # Convert to ms
        
        # 1. Accuracy comparison
        plt.figure(figsize=(12, 6))
        x = np.arange(len(variants))
        width = 0.35
        
        plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
        plt.bar(x + width/2, val_accs, width, label='Validation Accuracy')
        
        plt.xlabel('Model Variant')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison Across Model Variants')
        plt.xticks(x, variants, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_ablation_accuracy")
        save_figure(plt, base_filename, formats=['png', 'svg'])
        
        # 2. Parameters vs. Accuracy
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(params, val_accs, s=100, alpha=0.7)
        
        # Add variant names as annotations
        for i, variant in enumerate(variants):
            plt.annotate(
                variant, 
                (params[i], val_accs[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Number of Parameters')
        plt.ylabel('Validation Accuracy')
        plt.title('Model Size vs. Accuracy Trade-off')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_ablation_params_vs_acc")
        save_figure(plt, base_filename, formats=['png', 'svg'])
        
        # 3. Inference Time vs. Accuracy
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(inference_times, val_accs, s=100, alpha=0.7)
        
        # Add variant names as annotations
        for i, variant in enumerate(variants):
            plt.annotate(
                variant, 
                (inference_times[i], val_accs[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Validation Accuracy')
        plt.title('Inference Time vs. Accuracy Trade-off')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_ablation_inference_vs_acc")
        save_figure(plt, base_filename, formats=['png', 'svg'])
        
        # 4. Training curves for each variant
        plt.figure(figsize=(12, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training loss
        for r in self.results:
            ax1.plot(r['train_losses'], label=f"{r['variant_name']} (Train)")
            ax1.plot(r['val_losses'], linestyle='--', label=f"{r['variant_name']} (Val)")
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot training accuracy
        for r in self.results:
            ax2.plot(r['train_accs'], label=f"{r['variant_name']} (Train)")
            ax2.plot(r['val_accs'], linestyle='--', label=f"{r['variant_name']} (Val)")
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_ablation_training_curves")
        save_figure(plt, base_filename, formats=['png', 'svg'])


def create_model_variant(
    base_model: nn.Module, 
    modification_fn: Callable[[nn.Module], nn.Module],
    variant_name: str,
    description: str
) -> Dict[str, Any]:
    """
    Create a model variant for ablation study.
    
    Args:
        base_model: Base model to modify
        modification_fn: Function that takes a model and returns a modified version
        variant_name: Name for the variant
        description: Description of the modification
        
    Returns:
        Dictionary with variant information
    """
    # Create a deep copy of the base model
    variant_model = type(base_model)()  # Create a new instance
    variant_model.load_state_dict(base_model.state_dict())
    
    # Apply modification
    modified_model = modification_fn(variant_model)
    
    return {
        'name': variant_name,
        'model': modified_model,
        'description': description
    }


def remove_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Remove a layer from a model by replacing it with Identity.
    
    Args:
        model: Model to modify
        layer_name: Name of the layer to remove (e.g., 'conv1')
        
    Returns:
        Modified model
    """
    # This is a simple example - actual implementation depends on model structure
    if hasattr(model, layer_name):
        # Get the layer
        layer = getattr(model, layer_name)
        
        # Replace with Identity that preserves input shape
        if isinstance(layer, nn.Conv2d):
            setattr(model, layer_name, nn.Identity())
        elif isinstance(layer, nn.BatchNorm2d):
            setattr(model, layer_name, nn.Identity())
        elif isinstance(layer, nn.Linear):
            setattr(model, layer_name, nn.Identity())
    
    return model


def change_activation(model: nn.Module, activation_type: str) -> nn.Module:
    """
    Change all activation functions in a model.
    
    Args:
        model: Model to modify
        activation_type: Type of activation to use ('relu', 'leaky_relu', 'elu', etc.)
        
    Returns:
        Modified model
    """
    # Map activation type to class
    activation_map = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid
    }
    
    # Get activation class
    if activation_type not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    activation_class = activation_map[activation_type]
    
    # Replace all activation functions
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) or \
           isinstance(module, nn.ELU) or isinstance(module, nn.GELU) or \
           isinstance(module, nn.Tanh) or isinstance(module, nn.Sigmoid):
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, activation_class())
            else:
                setattr(model, child_name, activation_class())
    
    return model


def change_dropout_rate(model: nn.Module, dropout_rate: float) -> nn.Module:
    """
    Change all dropout rates in a model.
    
    Args:
        model: Model to modify
        dropout_rate: New dropout rate
        
    Returns:
        Modified model
    """
    # Replace all dropout layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Create new dropout with same type but different rate
                if isinstance(module, nn.Dropout):
                    setattr(parent, child_name, nn.Dropout(dropout_rate))
                else:  # nn.Dropout2d
                    setattr(parent, child_name, nn.Dropout2d(dropout_rate))
            else:
                if isinstance(module, nn.Dropout):
                    setattr(model, child_name, nn.Dropout(dropout_rate))
                else:  # nn.Dropout2d
                    setattr(model, child_name, nn.Dropout2d(dropout_rate))
    
    return model


def change_normalization(model: nn.Module, norm_type: str) -> nn.Module:
    """
    Change all normalization layers in a model.
    
    Args:
        model: Model to modify
        norm_type: Type of normalization to use ('batch', 'instance', 'layer', 'none')
        
    Returns:
        Modified model
    """
    # Replace all normalization layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                              nn.LayerNorm)):
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get num_features
            if hasattr(module, 'num_features'):
                num_features = module.num_features
            elif hasattr(module, 'normalized_shape'):
                num_features = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
            else:
                continue  # Skip if we can't determine num_features
            
            # Create new normalization layer
            if norm_type == 'batch':
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.LayerNorm):
                    new_norm = nn.BatchNorm1d(num_features)
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    new_norm = nn.BatchNorm2d(num_features)
                else:  # nn.BatchNorm3d or nn.InstanceNorm3d
                    new_norm = nn.BatchNorm3d(num_features)
            elif norm_type == 'instance':
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.LayerNorm):
                    new_norm = nn.InstanceNorm1d(num_features)
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    new_norm = nn.InstanceNorm2d(num_features)
                else:  # nn.BatchNorm3d or nn.InstanceNorm3d
                    new_norm = nn.InstanceNorm3d(num_features)
            elif norm_type == 'layer':
                new_norm = nn.LayerNorm(num_features)
            elif norm_type == 'none':
                new_norm = nn.Identity()
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
            
            # Set new normalization layer
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_norm)
            else:
                setattr(model, child_name, new_norm)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Ablation studies module loaded successfully.")
    print("Use AblationStudy class to perform ablation studies on model components.")
    print("Use create_model_variant() to create model variants for ablation studies.") 