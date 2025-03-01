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
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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
    
    try:
        # First, ensure the model is on CPU
        model = model.cpu()
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device='cpu')
        
        # Set file path
        export_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        # Start timer
        start_time = time.time()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        # Measure time
        export_time = time.time() - start_time
        
        # Measure size of exported file
        file_size = os.path.getsize(export_path) / (1024 * 1024)  # Size in MB
        
        return {
            'format': 'onnx',
            'path': export_path,
            'size_mb': file_size,
            'export_time': export_time
        }
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return {
            'format': 'onnx',
            'error': str(e),
            'size_mb': 0,
            'export_time': 0
        }


def export_to_torchscript(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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
    
    try:
        # First, ensure the model is on CPU
        model = model.cpu()
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device='cpu')
        
        # Set file path
        export_path = os.path.join(output_dir, f"{model_name}.pt")
        
        # Start timer
        start_time = time.time()
        
        # Export to TorchScript
        with torch.no_grad():
            traced_script_module = torch.jit.trace(model, dummy_input)
            traced_script_module.save(export_path)
        
        # Measure time
        export_time = time.time() - start_time
        
        # Measure size of exported file
        file_size = os.path.getsize(export_path) / (1024 * 1024)  # Size in MB
        
        return {
            'format': 'torchscript',
            'path': export_path,
            'size_mb': file_size,
            'export_time': export_time
        }
    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
        return {
            'format': 'torchscript',
            'error': str(e),
            'size_mb': 0,
            'export_time': 0
        }


def benchmark_deployment_metrics(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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
    deployment_dir = os.path.join(output_dir, 'deployment')
    os.makedirs(deployment_dir, exist_ok=True)
    
    try:
        # Make a copy of the model to avoid modifying the original
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        
        # Measure model size
        model_size = calculate_model_size(model_copy)
        
        # Count parameters
        params = count_parameters(model_copy)
        
        # Export to different formats - ensure model is on CPU for export
        model_copy = model_copy.cpu()
        onnx_metrics = export_to_onnx(model_copy, model_name, input_size, deployment_dir)
        torchscript_metrics = export_to_torchscript(model_copy, model_name, input_size, deployment_dir)
        
        # Measure latency for different batch sizes
        latency_results = []
        
        for batch_size in batch_sizes:
            # Adjust input size for current batch size
            current_input_size = (batch_size,) + input_size[1:]
            
            # Measure latency - try both CPU and CUDA if available
            try:
                # First try with CPU
                latency_metrics = measure_inference_latency(model_copy, current_input_size, device='cpu')
                
                latency_results.append({
                    'batch_size': batch_size,
                    'device': 'cpu',
                    **latency_metrics
                })
                
                # Try with CUDA if available
                if torch.cuda.is_available():
                    cuda_latency_metrics = measure_inference_latency(model_copy, current_input_size, device='cuda')
                    
                    latency_results.append({
                        'batch_size': batch_size,
                        'device': 'cuda',
                        **cuda_latency_metrics
                    })
            except Exception as e:
                print(f"Error measuring latency for batch size {batch_size}: {e}")
                latency_results.append({
                    'batch_size': batch_size,
                    'device': 'error',
                    'mean_latency': 0,
                    'throughput': 0,
                    'error': str(e)
                })
        
        # Create DataFrame
        latency_df = pd.DataFrame(latency_results)
        
        # Save latency results to CSV
        latency_csv = os.path.join(deployment_dir, f"{model_name}_latency.csv")
        latency_df.to_csv(latency_csv, index=False)
        
        # Create summary DataFrame
        summary_data = [{
            'Model': model_name,
            'Parameters': params,
            'Model Size (MB)': model_size,
            'ONNX Size (MB)': onnx_metrics.get('size_mb', 0),
            'TorchScript Size (MB)': torchscript_metrics.get('size_mb', 0),
            'Mean CPU Latency (ms)': latency_df[latency_df['device'] == 'cpu']['mean_latency'].mean() if not latency_df[latency_df['device'] == 'cpu'].empty else 0,
            'Mean GPU Latency (ms)': latency_df[latency_df['device'] == 'cuda']['mean_latency'].mean() if not latency_df[latency_df['device'] == 'cuda'].empty else 0,
            'Max CPU Throughput (samples/s)': latency_df[latency_df['device'] == 'cpu']['throughput'].max() if not latency_df[latency_df['device'] == 'cpu'].empty else 0,
            'Max GPU Throughput (samples/s)': latency_df[latency_df['device'] == 'cuda']['throughput'].max() if not latency_df[latency_df['device'] == 'cuda'].empty else 0
        }]
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_csv = os.path.join(deployment_dir, f"{model_name}_deployment_metrics.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        # Create visualizations
        # Visualize latency vs batch size for CPU and GPU
        plt.figure(figsize=(12, 6))
        
        # Filter data for plotting
        cpu_data = latency_df[latency_df['device'] == 'cpu']
        cuda_data = latency_df[latency_df['device'] == 'cuda']
        
        if not cpu_data.empty:
            plt.plot(cpu_data['batch_size'], cpu_data['mean_latency'], marker='o', label='CPU')
        
        if not cuda_data.empty:
            plt.plot(cuda_data['batch_size'], cuda_data['mean_latency'], marker='s', label='GPU')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Mean Latency (ms)')
        plt.title(f'Inference Latency vs Batch Size - {model_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(deployment_dir, f"{model_name}_latency_vs_batch_size")
        latency_plot = save_figure(plt, plot_path, formats=['png', 'svg'])
        
        return summary_df, summary_csv, latency_plot
    except Exception as e:
        print(f"Error in benchmark_deployment_metrics: {e}")
        # Return empty DataFrame
        return pd.DataFrame(), "", ""


def compare_deployment_metrics(
    models: List[Dict[str, Any]],
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
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