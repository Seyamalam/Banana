import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import csv
from typing import Dict, List, Tuple, Optional, Union, Any
from cell5_visualization import save_figure


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size(model: torch.nn.Module, save_path: Optional[str] = None) -> float:
    """
    Calculate the size of a PyTorch model in MB.
    
    Args:
        model: PyTorch model
        save_path: Path to save the model temporarily for size calculation
        
    Returns:
        Model size in MB
    """
    if save_path is None:
        save_path = "temp_model.pt"
        
    # Save model to disk
    torch.save(model.state_dict(), save_path)
    
    # Get file size
    size_bytes = os.path.getsize(save_path)
    size_mb = size_bytes / (1024 * 1024)
    
    # Remove temporary file
    if os.path.exists(save_path):
        os.remove(save_path)
        
    return size_mb


def calculate_advanced_efficiency_metrics(
    model_results: List[Dict[str, Any]],
    output_dir: str = "models"
) -> Tuple[pd.DataFrame, str, str]:
    """
    Calculate advanced efficiency metrics for model comparison.
    
    Args:
        model_results: List of dictionaries containing model results
                      Each dict should have keys: 'name', 'model', 'accuracy', 
                      'inference_time', 'training_time', 'model_size'
        output_dir: Directory to save results
        
    Returns:
        DataFrame with efficiency metrics, path to CSV file, and path to plot
    """
    # Initialize results
    results = []
    
    for result in model_results:
        model_name = result['name']
        model = result['model']
        accuracy = result['accuracy']
        inference_time = result['inference_time']  # in seconds
        training_time = result['training_time']    # in seconds
        model_size = result['model_size']          # in MB
        
        # Count parameters if not provided
        if 'parameters' not in result:
            parameters = count_parameters(model)
        else:
            parameters = result['parameters']
        
        # Calculate efficiency metrics
        accuracy_per_param = (accuracy * 100) / (parameters / 1_000_000)  # accuracy % per million params
        accuracy_per_mb = (accuracy * 100) / model_size                   # accuracy % per MB
        accuracy_per_inference_ms = (accuracy * 100) / (inference_time * 1000)  # accuracy % per ms
        accuracy_per_training_hour = (accuracy * 100) / (training_time / 3600)  # accuracy % per hour
        
        # Calculate combined efficiency score (higher is better)
        # Normalize each metric to 0-1 range within the dataset
        results.append({
            'name': model_name,
            'accuracy': accuracy,
            'parameters': parameters,
            'model_size': model_size,
            'inference_time': inference_time,
            'training_time': training_time,
            'accuracy_per_million_params': accuracy_per_param,
            'accuracy_per_mb': accuracy_per_mb,
            'accuracy_per_inference_ms': accuracy_per_inference_ms,
            'accuracy_per_training_hour': accuracy_per_training_hour
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate combined efficiency score
    # First normalize each metric (higher is better)
    metrics_to_normalize = [
        'accuracy_per_million_params',
        'accuracy_per_mb',
        'accuracy_per_inference_ms',
        'accuracy_per_training_hour'
    ]
    
    # Create normalized columns
    for metric in metrics_to_normalize:
        min_val = df[metric].min()
        max_val = df[metric].max()
        range_val = max_val - min_val
        
        if range_val > 0:
            df[f'{metric}_normalized'] = (df[metric] - min_val) / range_val
        else:
            df[f'{metric}_normalized'] = 1.0  # If all values are the same
    
    # Calculate combined score (equal weights)
    normalized_columns = [f'{metric}_normalized' for metric in metrics_to_normalize]
    df['efficiency_score'] = df[normalized_columns].mean(axis=1)
    
    # Sort by efficiency score
    df = df.sort_values('efficiency_score', ascending=False)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_efficiency_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # Create radar chart for top models
    top_n = min(5, len(df))
    top_models = df.head(top_n)
    
    # Prepare data for radar chart
    categories = ['Accuracy/Params', 'Accuracy/Size', 'Accuracy/Inference', 'Accuracy/Training']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Number of categories
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [
            row['accuracy_per_million_params_normalized'],
            row['accuracy_per_mb_normalized'],
            row['accuracy_per_inference_ms_normalized'],
            row['accuracy_per_training_hour_normalized']
        ]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['name'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Efficiency Comparison (Normalized Metrics)")
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_efficiency_radar")
    radar_png, radar_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # Create bar chart for efficiency score
    plt.figure(figsize=(12, 6))
    plt.bar(df['name'], df['efficiency_score'])
    plt.xlabel('Model')
    plt.ylabel('Efficiency Score')
    plt.title('Combined Efficiency Score (higher is better)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(df['efficiency_score']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_efficiency_score")
    score_png, score_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, radar_png


def calculate_pareto_frontier(
    model_results: List[Dict[str, Any]],
    x_metric: str = 'model_size',
    y_metric: str = 'accuracy',
    output_dir: str = "models"
) -> Tuple[pd.DataFrame, str, str]:
    """
    Calculate and visualize the Pareto frontier for model comparison.
    
    Args:
        model_results: List of dictionaries containing model results
        x_metric: Metric for x-axis (lower is better, e.g., model_size, inference_time)
        y_metric: Metric for y-axis (higher is better, e.g., accuracy)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with Pareto optimal models, path to CSV file, and path to plot
    """
    # Convert to DataFrame
    df = pd.DataFrame(model_results)
    
    # Identify Pareto optimal points
    pareto_optimal = np.ones(len(df), dtype=bool)
    
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                # Check if point j dominates point i
                if (row_j[x_metric] <= row_i[x_metric] and 
                    row_j[y_metric] >= row_i[y_metric] and
                    (row_j[x_metric] < row_i[x_metric] or 
                     row_j[y_metric] > row_i[y_metric])):
                    pareto_optimal[i] = False
                    break
    
    # Add Pareto optimality to DataFrame
    df['pareto_optimal'] = pareto_optimal
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"pareto_frontier_{x_metric}_vs_{y_metric}.csv")
    df.to_csv(csv_path, index=False)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot all models
    plt.scatter(
        df[~df['pareto_optimal']][x_metric], 
        df[~df['pareto_optimal']][y_metric], 
        alpha=0.7, 
        label='Dominated Models'
    )
    
    # Plot Pareto optimal models
    pareto_df = df[df['pareto_optimal']]
    plt.scatter(
        pareto_df[x_metric], 
        pareto_df[y_metric], 
        color='red', 
        s=100, 
        alpha=0.7, 
        label='Pareto Optimal Models'
    )
    
    # Connect Pareto optimal points
    pareto_df = pareto_df.sort_values(x_metric)
    plt.plot(
        pareto_df[x_metric], 
        pareto_df[y_metric], 
        'r--', 
        alpha=0.7
    )
    
    # Add model names as annotations
    for _, row in df.iterrows():
        plt.annotate(
            row['name'], 
            (row[x_metric], row[y_metric]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add labels and title
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    plt.title(f'Pareto Frontier: {y_metric.title()} vs {x_metric.title()}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    base_filename = os.path.join(output_dir, f"pareto_frontier_{x_metric}_vs_{y_metric}")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return pareto_df, csv_path, png_path


def calculate_efficiency_frontier(
    model_results: List[Dict[str, Any]],
    output_dir: str = "models"
) -> Dict[str, Tuple[pd.DataFrame, str, str]]:
    """
    Calculate efficiency frontiers for multiple metric combinations.
    
    Args:
        model_results: List of dictionaries containing model results
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping metric pairs to (DataFrame, csv_path, plot_path) tuples
    """
    # Define metric pairs (x_metric: lower is better, y_metric: higher is better)
    metric_pairs = [
        ('model_size', 'accuracy'),
        ('inference_time', 'accuracy'),
        ('parameters', 'accuracy'),
        ('training_time', 'accuracy')
    ]
    
    results = {}
    
    for x_metric, y_metric in metric_pairs:
        pareto_df, csv_path, plot_path = calculate_pareto_frontier(
            model_results, 
            x_metric=x_metric, 
            y_metric=y_metric,
            output_dir=output_dir
        )
        results[f'{x_metric}_vs_{y_metric}'] = (pareto_df, csv_path, plot_path)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Efficiency metrics module loaded successfully.")
    print("Use calculate_advanced_efficiency_metrics() to compute efficiency metrics.")
    print("Use calculate_pareto_frontier() to identify Pareto optimal models.")
    print("Use calculate_efficiency_frontier() to analyze multiple efficiency trade-offs.") 