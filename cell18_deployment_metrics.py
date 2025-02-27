import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
import io
from typing import Dict, List, Tuple, Optional, Union, Any
from cell5_visualization import save_figure
from cell13_efficiency_metrics import count_parameters, calculate_model_size


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure inference latency for a PyTorch model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run measurement on
        num_iterations: Number of iterations to measure
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dictionary with latency metrics
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            # Synchronize GPU to get accurate timing
            if device == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.time() - start_time)
    
    # Calculate statistics
    latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds
    avg_latency = np.mean(latencies_ms)
    std_latency = np.std(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    p95_latency = np.percentile(latencies_ms, 95)
    p99_latency = np.percentile(latencies_ms, 99)
    
    return {
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency
    }


def export_to_onnx(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    output_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        input_size: Input tensor size (batch_size, channels, height, width)
        output_dir: Directory to save exported model
        
    Returns:
        Dictionary with export metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Get file size
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # Convert to MB
    
    return {
        'model_name': model_name,
        'format': 'onnx',
        'file_path': onnx_path,
        'size_mb': onnx_size
    }


def export_to_torchscript(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    output_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        input_size: Input tensor size (batch_size, channels, height, width)
        output_dir: Directory to save exported model
        
    Returns:
        Dictionary with export metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    script_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.jit.save(traced_model, script_path)
    
    # Get file size
    script_size = os.path.getsize(script_path) / (1024 * 1024)  # Convert to MB
    
    return {
        'model_name': model_name,
        'format': 'torchscript',
        'file_path': script_path,
        'size_mb': script_size
    }


def benchmark_deployment_metrics(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Benchmark deployment metrics for a PyTorch model.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        input_size: Input tensor size (batch_size, channels, height, width)
        batch_sizes: List of batch sizes to benchmark
        output_dir: Directory to save results
        
    Returns:
        DataFrame with benchmark results, path to CSV file, and path to plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Measure model size
    model_size = calculate_model_size(model)
    
    # Count parameters
    params = count_parameters(model)
    
    # Export to different formats
    onnx_metrics = export_to_onnx(model, model_name, input_size, output_dir)
    torchscript_metrics = export_to_torchscript(model, model_name, input_size, output_dir)
    
    # Measure latency for different batch sizes
    latency_results = []
    
    for batch_size in batch_sizes:
        # Adjust input size for current batch size
        current_input_size = (batch_size,) + input_size[1:]
        
        # Measure latency
        latency_metrics = measure_inference_latency(model, current_input_size)
        
        # Calculate throughput (images/second)
        throughput = batch_size * 1000 / latency_metrics['avg_latency_ms']
        
        latency_results.append({
            'batch_size': batch_size,
            'avg_latency_ms': latency_metrics['avg_latency_ms'],
            'throughput_imgs_per_sec': throughput
        })
    
    # Create DataFrame for latency results
    latency_df = pd.DataFrame(latency_results)
    
    # Save latency results to CSV
    latency_csv_path = os.path.join(output_dir, f"{model_name}_latency_benchmark.csv")
    latency_df.to_csv(latency_csv_path, index=False)
    
    # Create summary DataFrame
    summary_data = {
        'model_name': model_name,
        'parameters': params,
        'pytorch_model_size_mb': model_size,
        'onnx_model_size_mb': onnx_metrics['size_mb'],
        'torchscript_model_size_mb': torchscript_metrics['size_mb'],
        'latency_batch1_ms': latency_results[0]['avg_latency_ms'],
        'max_throughput_imgs_per_sec': max([r['throughput_imgs_per_sec'] for r in latency_results])
    }
    
    summary_df = pd.DataFrame([summary_data])
    
    # Save summary to CSV
    summary_csv_path = os.path.join(output_dir, f"{model_name}_deployment_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create visualizations
    
    # 1. Latency vs Batch Size
    plt.figure(figsize=(10, 6))
    plt.plot(latency_df['batch_size'], latency_df['avg_latency_ms'], marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Latency (ms)')
    plt.title(f'Inference Latency vs Batch Size - {model_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    base_filename = os.path.join(output_dir, f"{model_name}_latency_vs_batch_size")
    latency_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 2. Throughput vs Batch Size
    plt.figure(figsize=(10, 6))
    plt.plot(latency_df['batch_size'], latency_df['throughput_imgs_per_sec'], marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (images/second)')
    plt.title(f'Inference Throughput vs Batch Size - {model_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    base_filename = os.path.join(output_dir, f"{model_name}_throughput_vs_batch_size")
    throughput_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 3. Model Size Comparison
    plt.figure(figsize=(10, 6))
    sizes = [model_size, onnx_metrics['size_mb'], torchscript_metrics['size_mb']]
    labels = ['PyTorch', 'ONNX', 'TorchScript']
    
    plt.bar(labels, sizes)
    plt.xlabel('Format')
    plt.ylabel('Model Size (MB)')
    plt.title(f'Model Size Comparison - {model_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(sizes):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    # Save figure
    base_filename = os.path.join(output_dir, f"{model_name}_model_size_comparison")
    size_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return summary_df, summary_csv_path, latency_png


def compare_deployment_metrics(
    models: List[Dict[str, Any]],
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Compare deployment metrics across multiple models.
    
    Args:
        models: List of dictionaries, each containing:
               - 'name': Model name
               - 'model': PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results, path to CSV file, and path to plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark each model
    all_results = []
    
    for model_dict in models:
        model_name = model_dict['name']
        model = model_dict['model']
        
        print(f"Benchmarking deployment metrics for {model_name}...")
        
        # Measure model size
        model_size = calculate_model_size(model)
        
        # Count parameters
        params = count_parameters(model)
        
        # Measure latency
        latency_metrics = measure_inference_latency(model, input_size)
        
        # Calculate throughput (images/second)
        throughput = input_size[0] * 1000 / latency_metrics['avg_latency_ms']
        
        # Store results
        all_results.append({
            'model_name': model_name,
            'parameters': params,
            'model_size_mb': model_size,
            'avg_latency_ms': latency_metrics['avg_latency_ms'],
            'p95_latency_ms': latency_metrics['p95_latency_ms'],
            'throughput_imgs_per_sec': throughput
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "model_deployment_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Create visualizations
    
    # 1. Latency Comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['avg_latency_ms'])
    plt.xlabel('Model')
    plt.ylabel('Average Latency (ms)')
    plt.title('Inference Latency Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(df['avg_latency_ms']):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_latency_comparison")
    latency_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 2. Throughput Comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['throughput_imgs_per_sec'])
    plt.xlabel('Model')
    plt.ylabel('Throughput (images/second)')
    plt.title('Inference Throughput Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(df['throughput_imgs_per_sec']):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_throughput_comparison")
    throughput_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 3. Size vs Latency Scatter Plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(df['model_size_mb'], df['avg_latency_ms'], s=100, alpha=0.7)
    
    # Add model names as annotations
    for i, row in df.iterrows():
        plt.annotate(
            row['model_name'], 
            (row['model_size_mb'], row['avg_latency_ms']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Average Latency (ms)')
    plt.title('Model Size vs Inference Latency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_size_vs_latency")
    scatter_png, _ = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, latency_png


if __name__ == "__main__":
    # Example usage
    print("Deployment metrics module loaded successfully.")
    print("Use benchmark_deployment_metrics() to measure deployment metrics for a model.")
    print("Use compare_deployment_metrics() to compare deployment metrics across models.") 