import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import json
import torch.nn as nn
from tqdm import tqdm
import csv

from cell1_imports_and_constants import CLASS_NAMES, IDX_TO_CLASS, NUM_CLASSES
from cell8_model_zoo import ModelInfo

class ModelComparer:
    """
    Class to handle model comparison, evaluation and visualization
    """
    def __init__(self, results_dir="models/comparison_results"):
        """
        Initialize the model comparer
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories
        self.vis_dir = os.path.join(results_dir, "visualizations")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        self.models_info_dir = os.path.join(results_dir, "models_info")
        
        for directory in [self.vis_dir, self.metrics_dir, self.models_info_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.models_info = []
        self.all_metrics = {}
        self.evaluation_results = {}
        
    def add_model_info(self, model_info):
        """
        Add model information for comparison
        """
        self.models_info.append(model_info)
        
    def load_model_infos(self, models_dir="models"):
        """
        Load model information from saved results
        """
        # Look for model_info.json files in subdirectories
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file == "model_summary.json":
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            data = json.load(f)
                            
                            # Extract model name from directory structure
                            model_name = os.path.basename(root)
                            
                            # Look for evaluation metrics
                            eval_file = os.path.join(root, '..', 'evaluation', 'evaluation_metrics.csv')
                            if os.path.exists(eval_file):
                                metrics = pd.read_csv(eval_file)
                                accuracy = metrics[metrics['metric'] == 'test_accuracy']['value'].values[0]
                                loss = metrics[metrics['metric'] == 'test_loss']['value'].values[0]
                                inference_time = metrics[metrics['metric'] == 'inference_time_seconds']['value'].values[0]
                            else:
                                accuracy = None
                                loss = None
                                inference_time = None
                            
                            # Look for class-wise metrics
                            class_file = os.path.join(root, '..', 'evaluation', 'class_metrics.csv')
                            if os.path.exists(class_file):
                                class_metrics = pd.read_csv(class_file)
                                class_accuracies = dict(zip(class_metrics['class'], class_metrics['accuracy']))
                            else:
                                class_accuracies = None
                            
                            # Create ModelInfo object
                            model_info = ModelInfo(
                                model_name=data.get('model_name', model_name),
                                model_type='custom',  # Assume custom for now
                                params=data.get('total_parameters', 0),
                                trainable_params=data.get('trainable_parameters', 0),
                                model_size_mb=data.get('model_size_mb', 0),
                                accuracy=accuracy,
                                loss=loss,
                                inference_time=inference_time,
                                class_accuracies=class_accuracies
                            )
                            
                            self.add_model_info(model_info)
                            print(f"Loaded model info for {model_info.model_name}")
                    except Exception as e:
                        print(f"Error loading model info: {e}")
    
    def evaluate_model(self, model, test_loader, criterion, device, model_name=None, transforms=None):
        """
        Evaluate a model on the test dataset
        
        Args:
            model: The model to evaluate
            test_loader: DataLoader for test data
            criterion: Loss function
            device: Device to run evaluation on
            model_name: Name of the model (for tracking)
            transforms: Optional transforms for this specific model
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        start_time = time.time()
        
        # Use custom transform if provided
        if transforms is not None:
            print(f"Using custom transforms for {model_name}")
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                # Apply custom transforms if needed
                if transforms is not None:
                    # We need to convert the tensor back to PIL images
                    from PIL import Image
                    import torchvision.transforms.functional as F
                    
                    transformed_inputs = []
                    for img in inputs:
                        # Convert to PIL
                        pil_img = F.to_pil_image(img)
                        # Apply transforms
                        transformed_img = transforms(pil_img)
                        transformed_inputs.append(transformed_img)
                    
                    # Stack back to a batch
                    inputs = torch.stack(transformed_inputs)
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Collect metrics
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        inference_time = time.time() - start_time
        total_samples = len(all_targets)
        avg_inference_time = inference_time / total_samples
        
        test_loss = test_loss / total_samples
        correct = sum(1 for i in range(len(all_preds)) if all_preds[i] == all_targets[i])
        test_acc = 100. * correct / total_samples
        
        # Calculate per-class metrics
        class_accuracies = {}
        for i in range(NUM_CLASSES):
            # Get indices for this class
            indices = [j for j, label in enumerate(all_targets) if label == i]
            if indices:
                class_correct = sum(1 for j in indices if all_preds[j] == all_targets[j])
                class_accuracies[IDX_TO_CLASS[i]] = 100. * class_correct / len(indices)
            else:
                class_accuracies[IDX_TO_CLASS[i]] = 0.0
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Store metrics
        metrics = {
            'model_name': model_name,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'total_samples': total_samples,
            'correct': correct,
            'inference_time': inference_time,
            'avg_inference_time': avg_inference_time,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm,
            'all_preds': all_preds,
            'all_targets': all_targets
        }
        
        # Update model info
        if model_name:
            for model_info in self.models_info:
                if model_info.model_name == model_name:
                    model_info.accuracy = test_acc
                    model_info.loss = test_loss
                    model_info.inference_time = avg_inference_time
                    model_info.class_accuracies = class_accuracies
        
        # Store in evaluation results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_models(self, models, test_loader, device):
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of models with their transforms {name: (model, transforms)}
            test_loader: DataLoader for test data
            device: Device to run evaluation on
        """
        criterion = nn.CrossEntropyLoss()
        
        for name, (model, transforms) in models.items():
            print(f"Evaluating model: {name}")
            metrics = self.evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                model_name=name,
                transforms=transforms
            )
            
            # Save metrics
            results_path = os.path.join(self.metrics_dir, f"{name}_metrics.json")
            with open(results_path, 'w') as f:
                # Convert NumPy arrays to lists for JSON serialization
                metrics_json = metrics.copy()
                metrics_json['confusion_matrix'] = metrics_json['confusion_matrix'].tolist()
                metrics_json['all_preds'] = metrics_json['all_preds'].tolist()
                metrics_json['all_targets'] = metrics_json['all_targets'].tolist()
                json.dump(metrics_json, f, indent=4)
            
            print(f"Results for {name}:")
            print(f"  Accuracy: {metrics['test_acc']:.2f}%")
            print(f"  Loss: {metrics['test_loss']:.4f}")
            print(f"  Inference time: {metrics['avg_inference_time']*1000:.2f} ms per sample")
    
    def save_all_model_info(self):
        """
        Save all model information to CSV and JSON
        """
        # Convert to DataFrame for easy CSV export
        data = [model_info.to_dict() for model_info in self.models_info]
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(self.models_info_dir, "all_models_info.csv")
        df.to_csv(csv_path, index=False)
        print(f"All model info saved to {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.models_info_dir, "all_models_info.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"All model info saved to {json_path}")
    
    def visualize_model_comparison(self, metrics=None, save_formats=None):
        """
        Visualize model comparisons
        
        Args:
            metrics: List of metrics to visualize, e.g., ['accuracy', 'loss', 'inference_time']
            save_formats: List of formats to save visualizations in, e.g., ['png', 'svg']
        """
        if metrics is None:
            metrics = ['accuracy', 'params', 'size', 'inference_time']
        
        if save_formats is None:
            save_formats = ['png', 'svg']
        
        # Extract data
        model_names = [info.model_name for info in self.models_info]
        model_types = [info.model_type for info in self.models_info]
        params = [info.params for info in self.models_info]
        trainable_params = [info.trainable_params for info in self.models_info]
        model_sizes = [info.model_size_mb for info in self.models_info]
        accuracies = [info.accuracy for info in self.models_info if info.accuracy is not None]
        accuracies_models = [info.model_name for info in self.models_info if info.accuracy is not None]
        inference_times = [info.inference_time * 1000 for info in self.models_info if info.inference_time is not None]  # Convert to ms
        inference_times_models = [info.model_name for info in self.models_info if info.inference_time is not None]
        
        # Visualization 1: Model size vs. parameters (bubble chart with accuracy)
        if 'size' in metrics and 'params' in metrics:
            plt.figure(figsize=(10, 8))
            
            # Split by model type
            custom_indices = [i for i, t in enumerate(model_types) if t == 'custom']
            pretrained_indices = [i for i, t in enumerate(model_types) if t == 'pretrained']
            
            # Plot custom models
            if custom_indices:
                custom_sizes = [model_sizes[i] for i in custom_indices]
                custom_params = [params[i] for i in custom_indices]
                custom_names = [model_names[i] for i in custom_indices]
                custom_accuracies = [info.accuracy for i, info in enumerate(self.models_info) 
                                    if i in custom_indices and info.accuracy is not None]
                
                # Use default size if accuracy is not available
                sizes = [acc*20 if acc is not None else 200 for acc in custom_accuracies]
                
                plt.scatter(custom_sizes, custom_params, s=sizes, alpha=0.7, label='Custom Models')
                
                # Add labels
                for i, name in enumerate(custom_names):
                    plt.annotate(name, (custom_sizes[i], custom_params[i]), 
                                fontsize=8, alpha=0.8, ha='center')
            
            # Plot pretrained models
            if pretrained_indices:
                pretrained_sizes = [model_sizes[i] for i in pretrained_indices]
                pretrained_params = [params[i] for i in pretrained_indices]
                pretrained_names = [model_names[i] for i in pretrained_indices]
                pretrained_accuracies = [info.accuracy for i, info in enumerate(self.models_info) 
                                        if i in pretrained_indices and info.accuracy is not None]
                
                # Use default size if accuracy is not available
                sizes = [acc*20 if acc is not None else 200 for acc in pretrained_accuracies]
                
                plt.scatter(pretrained_sizes, pretrained_params, s=sizes, alpha=0.7, marker='s', 
                            label='Pre-trained Models')
                
                # Add labels
                for i, name in enumerate(pretrained_names):
                    plt.annotate(name, (pretrained_sizes[i], pretrained_params[i]), 
                                fontsize=8, alpha=0.8, ha='center')
            
            plt.xlabel('Model Size (MB)')
            plt.ylabel('Number of Parameters')
            plt.title('Model Size vs Parameters')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Log scale for better visualization
            plt.xscale('log')
            plt.yscale('log')
            
            # Save in multiple formats
            for fmt in save_formats:
                save_path = os.path.join(self.vis_dir, f"model_size_vs_params.{fmt}")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.close()
        
        # Visualization 2: Accuracy comparison
        if 'accuracy' in metrics and accuracies:
            plt.figure(figsize=(12, 6))
            
            # Sort by accuracy
            sorted_indices = np.argsort(accuracies)[::-1]  # Descending
            sorted_models = [accuracies_models[i] for i in sorted_indices]
            sorted_accuracies = [accuracies[i] for i in sorted_indices]
            
            # Get model types for coloring
            sorted_types = []
            for model in sorted_models:
                for info in self.models_info:
                    if info.model_name == model:
                        sorted_types.append(info.model_type)
                        break
            
            # Create color map
            colors = ['#3498db' if t == 'custom' else '#e74c3c' for t in sorted_types]
            
            bars = plt.bar(sorted_models, sorted_accuracies, color=colors)
            
            # Add value labels on top of bars
            for bar, acc in zip(bars, sorted_accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Model')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracy Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', label='Custom Models'),
                Patch(facecolor='#e74c3c', label='Pre-trained Models')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in save_formats:
                save_path = os.path.join(self.vis_dir, f"model_accuracy_comparison.{fmt}")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.close()
        
        # Visualization 3: Inference time comparison
        if 'inference_time' in metrics and inference_times:
            plt.figure(figsize=(12, 6))
            
            # Sort by inference time (ascending)
            sorted_indices = np.argsort(inference_times)
            sorted_models = [inference_times_models[i] for i in sorted_indices]
            sorted_times = [inference_times[i] for i in sorted_indices]
            
            # Get model types for coloring
            sorted_types = []
            for model in sorted_models:
                for info in self.models_info:
                    if info.model_name == model:
                        sorted_types.append(info.model_type)
                        break
            
            # Create color map
            colors = ['#3498db' if t == 'custom' else '#e74c3c' for t in sorted_types]
            
            bars = plt.bar(sorted_models, sorted_times, color=colors)
            
            # Add value labels on top of bars
            for bar, time in zip(bars, sorted_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time:.1f} ms', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Model')
            plt.ylabel('Inference Time (ms per sample)')
            plt.title('Model Inference Time Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', label='Custom Models'),
                Patch(facecolor='#e74c3c', label='Pre-trained Models')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in save_formats:
                save_path = os.path.join(self.vis_dir, f"model_inference_time_comparison.{fmt}")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.close()
        
        # Visualization 4: Per-class accuracy comparison
        models_with_class_acc = []
        class_acc_data = []
        
        for info in self.models_info:
            if info.class_accuracies is not None:
                models_with_class_acc.append(info.model_name)
                class_acc_data.append(info.class_accuracies)
        
        if models_with_class_acc:
            # Create a DataFrame with class accuracies
            class_acc_df = pd.DataFrame(class_acc_data, index=models_with_class_acc)
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(class_acc_df, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': 'Accuracy (%)'})
            plt.title('Per-Class Accuracy Comparison')
            plt.ylabel('Model')
            plt.xlabel('Class')
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in save_formats:
                save_path = os.path.join(self.vis_dir, f"per_class_accuracy_comparison.{fmt}")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            
            plt.close()
            
            # Also save as CSV
            csv_path = os.path.join(self.metrics_dir, "per_class_accuracy_comparison.csv")
            class_acc_df.to_csv(csv_path)
            print(f"Saved per-class accuracy data to {csv_path}")
        
        # Visualization 5: Model efficiency (accuracy/parameter count)
        if accuracies:
            # Calculate efficiency for models with accuracy
            efficiencies = []
            eff_model_names = []
            params_list = []
            
            for info in self.models_info:
                if info.accuracy is not None and info.params > 0:
                    efficiency = info.accuracy / np.log10(info.params)  # Use log scale for params
                    efficiencies.append(efficiency)
                    eff_model_names.append(info.model_name)
                    params_list.append(info.params)
            
            if efficiencies:
                plt.figure(figsize=(12, 6))
                
                # Sort by efficiency (descending)
                sorted_indices = np.argsort(efficiencies)[::-1]
                sorted_models = [eff_model_names[i] for i in sorted_indices]
                sorted_efficiencies = [efficiencies[i] for i in sorted_indices]
                sorted_params = [params_list[i] for i in sorted_indices]
                
                # Get model types for coloring
                sorted_types = []
                for model in sorted_models:
                    for info in self.models_info:
                        if info.model_name == model:
                            sorted_types.append(info.model_type)
                            break
                
                # Create color map
                colors = ['#3498db' if t == 'custom' else '#e74c3c' for t in sorted_types]
                
                bars = plt.bar(sorted_models, sorted_efficiencies, color=colors)
                
                # Add value labels on top of bars
                for bar, eff, params in zip(bars, sorted_efficiencies, sorted_params):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{eff:.1f}\n({params:,} params)', ha='center', va='bottom', fontsize=8)
                
                plt.xlabel('Model')
                plt.ylabel('Efficiency (Accuracy / log10(Parameters))')
                plt.title('Model Efficiency Comparison')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#3498db', label='Custom Models'),
                    Patch(facecolor='#e74c3c', label='Pre-trained Models')
                ]
                plt.legend(handles=legend_elements)
                
                plt.tight_layout()
                
                # Save in multiple formats
                for fmt in save_formats:
                    save_path = os.path.join(self.vis_dir, f"model_efficiency_comparison.{fmt}")
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Saved visualization to {save_path}")
                
                plt.close()

def generate_model_comparison_report(models_info, output_file):
    """
    Generate a markdown report comparing models
    
    Args:
        models_info: List of ModelInfo objects
        output_file: Path to save the report
    """
    with open(output_file, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        
        # Basic model information table
        f.write("## Model Information\n\n")
        f.write("| Model | Type | Parameters | Trainable Parameters | Size (MB) |\n")
        f.write("|-------|------|------------|----------------------|----------|\n")
        
        for info in models_info:
            f.write(f"| {info.model_name} | {info.model_type.capitalize()} | {info.params:,} | {info.trainable_params:,} | {info.model_size_mb:.2f} |\n")
        
        f.write("\n")
        
        # Performance metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Accuracy (%) | Loss | Inference Time (ms) |\n")
        f.write("|-------|-------------|------|---------------------|\n")
        
        for info in models_info:
            accuracy = f"{info.accuracy:.2f}" if info.accuracy is not None else "N/A"
            loss = f"{info.loss:.4f}" if info.loss is not None else "N/A"
            inf_time = f"{info.inference_time*1000:.2f}" if info.inference_time is not None else "N/A"
            
            f.write(f"| {info.model_name} | {accuracy} | {loss} | {inf_time} |\n")
        
        f.write("\n")
        
        # Class-wise accuracy section
        f.write("## Class-wise Accuracy\n\n")
        
        # First, collect all models with class accuracies
        models_with_class_acc = []
        for info in models_info:
            if info.class_accuracies is not None:
                models_with_class_acc.append(info)
        
        if models_with_class_acc:
            # Get all class names
            all_classes = list(models_with_class_acc[0].class_accuracies.keys())
            
            # Create table header
            f.write("| Model | " + " | ".join(all_classes) + " |\n")
            f.write("|-------|" + "-|".join(["-"*len(c) for c in all_classes]) + "-|\n")
            
            # Add data rows
            for info in models_with_class_acc:
                row = f"| {info.model_name} |"
                for cls in all_classes:
                    if cls in info.class_accuracies:
                        row += f" {info.class_accuracies[cls]:.2f} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")
        else:
            f.write("No class-wise accuracy data available.\n")
        
        f.write("\n")
        
        # Model efficiency section
        f.write("## Model Efficiency\n\n")
        f.write("Efficiency is calculated as Accuracy / log10(Parameters) to measure the trade-off between model performance and complexity.\n\n")
        f.write("| Model | Efficiency |\n")
        f.write("|-------|------------|\n")
        
        for info in models_info:
            if info.accuracy is not None and info.params > 0:
                efficiency = info.accuracy / np.log10(info.params)
                f.write(f"| {info.model_name} | {efficiency:.2f} |\n")
            else:
                f.write(f"| {info.model_name} | N/A |\n")
        
        f.write("\n")
        
        # Recommendations section
        f.write("## Recommendations\n\n")
        
        # Find best model for accuracy
        best_acc_model = None
        best_acc = -1
        
        # Find best model for efficiency
        best_eff_model = None
        best_eff = -1
        
        # Find best model for inference speed
        best_speed_model = None
        best_speed = float('inf')
        
        for info in models_info:
            # Accuracy
            if info.accuracy is not None and info.accuracy > best_acc:
                best_acc = info.accuracy
                best_acc_model = info.model_name
            
            # Efficiency
            if info.accuracy is not None and info.params > 0:
                efficiency = info.accuracy / np.log10(info.params)
                if efficiency > best_eff:
                    best_eff = efficiency
                    best_eff_model = info.model_name
            
            # Speed
            if info.inference_time is not None and info.inference_time < best_speed:
                best_speed = info.inference_time
                best_speed_model = info.model_name
        
        f.write("Based on the evaluation results, here are recommendations for different use cases:\n\n")
        
        if best_acc_model:
            f.write(f"- **Best Accuracy**: {best_acc_model} ({best_acc:.2f}%) - Recommended for applications where prediction accuracy is the primary concern.\n")
        
        if best_eff_model:
            f.write(f"- **Best Efficiency**: {best_eff_model} (Efficiency: {best_eff:.2f}) - Recommended for balanced accuracy and model complexity.\n")
        
        if best_speed_model:
            f.write(f"- **Fastest Inference**: {best_speed_model} ({best_speed*1000:.2f} ms per sample) - Recommended for real-time applications or resource-constrained environments.\n")
    
    print(f"Report generated at {output_file}")

def compare_models_with_pretrained(custom_model_dir, results_dir, test_loader, device, 
                                 pretrained_models=None, save_formats=None):
    """
    Compare custom models with pre-trained models
    
    Args:
        custom_model_dir: Directory containing custom model results
        results_dir: Directory to save comparison results
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        pretrained_models: Dictionary of pre-trained models {name: (model, transforms)}
        save_formats: List of formats to save visualizations (e.g., ['png', 'svg'])
    """
    # Set default values
    if save_formats is None:
        save_formats = ['png', 'svg']
    
    # Initialize model comparer
    comparer = ModelComparer(results_dir=results_dir)
    
    # Load custom model information
    comparer.load_model_infos(models_dir=custom_model_dir)
    
    # Evaluate pre-trained models if provided
    if pretrained_models is not None:
        # Add pre-trained model info
        for name, (model, _) in pretrained_models.items():
            model_info = get_model_info_from_model(model, model_type='pretrained')
            model_info.model_name = name  # Ensure name matches
            comparer.add_model_info(model_info)
        
        # Evaluate pre-trained models
        comparer.evaluate_models(pretrained_models, test_loader, device)
    
    # Save model information
    comparer.save_all_model_info()
    
    # Generate visualizations
    comparer.visualize_model_comparison(save_formats=save_formats)
    
    # Generate report
    report_path = os.path.join(results_dir, "model_comparison_report.md")
    generate_model_comparison_report(comparer.models_info, report_path)
    
    return comparer 