import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from cell5_visualization import save_figure
from thop import profile, clever_format
from ptflops import get_model_complexity_info
import time


def calculate_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Calculate FLOPs (Floating Point Operations) for a PyTorch model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run calculation on
        
    Returns:
        Dictionary with FLOPs and parameters information
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Calculate FLOPs using thop
    macs, params = profile(model, inputs=(dummy_input,))
    flops = macs * 2  # Multiply MACs by 2 to get FLOPs
    
    # Format for readability
    flops_readable, params_readable = clever_format([flops, params], "%.3f")
    
    # Calculate FLOPs per parameter
    flops_per_param = flops / params if params > 0 else 0
    
    # Calculate theoretical throughput (images/second) based on FLOPs
    # Assuming 10 TFLOPS for a modern GPU (adjust as needed)
    theoretical_throughput_gpu = 10e12 / flops if flops > 0 else 0
    
    # Assuming 100 GFLOPS for a modern CPU (adjust as needed)
    theoretical_throughput_cpu = 100e9 / flops if flops > 0 else 0
    
    # Measure actual throughput
    model.eval()
    batch_size = input_size[0]
    num_batches = 100
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure throughput
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            _ = model(dummy_input)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_throughput = (num_batches * batch_size) / elapsed_time
    
    return {
        'flops': flops,
        'flops_readable': flops_readable,
        'params': params,
        'params_readable': params_readable,
        'flops_per_param': flops_per_param,
        'theoretical_throughput_gpu': theoretical_throughput_gpu,
        'theoretical_throughput_cpu': theoretical_throughput_cpu,
        'actual_throughput': actual_throughput,
        'efficiency_ratio': actual_throughput / theoretical_throughput_gpu if device == 'cuda' else actual_throughput / theoretical_throughput_cpu
    }


def calculate_layer_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Calculate FLOPs for each layer in the model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run calculation on
        
    Returns:
        DataFrame with layer-wise FLOPs information
    """
    model = model.to(device)
    model.eval()
    
    try:
        # Get layer-wise complexity using ptflops
        macs_dict, params_dict = get_model_complexity_info(
            model, 
            input_size[1:],  # Remove batch dimension
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        
        # Check if macs_dict is a dictionary
        if not isinstance(macs_dict, dict):
            # Instead of showing a warning, provide a more informative message
            print(f"📊 Creating layer summary with total FLOPs for {model.__class__.__name__} - detailed layer breakdown not available")
            # Create a single entry for total MACs
            macs_dict = {"total": macs_dict}
            params_dict = {"total": params_dict}
        
        # Convert to DataFrame
        layers = []
        for layer_name, macs in macs_dict.items():
            params = params_dict.get(layer_name, 0)
            flops = macs * 2  # Multiply MACs by 2 to get FLOPs
            
            layers.append({
                'layer_name': layer_name,
                'flops': flops,
                'params': params,
                'flops_per_param': flops / params if params > 0 else 0
            })
        
        df = pd.DataFrame(layers)
        
        # Calculate percentages
        total_flops = df['flops'].sum()
        total_params = df['params'].sum()
        
        df['flops_percentage'] = (df['flops'] / total_flops * 100) if total_flops > 0 else 0
        df['params_percentage'] = (df['params'] / total_params * 100) if total_params > 0 else 0
        
        # Sort by FLOPs (descending)
        df = df.sort_values('flops', ascending=False)
        
        return df
    except Exception as e:
        print(f"Error in calculate_layer_flops: {e}")
        return pd.DataFrame()


def compare_model_flops(
    models: List[Dict[str, Any]],
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Compare FLOPs and computational efficiency across multiple models.
    
    Args:
        models: List of dictionaries, each containing:
               - 'name': Model name
               - 'model': PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run calculation on
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results, path to CSV file, and path to plot
    """
    # Calculate FLOPs for each model
    results = []
    
    for model_dict in models:
        model_name = model_dict['name']
        model = model_dict['model']
        
        print(f"Calculating FLOPs for {model_name}...")
        flops_info = calculate_flops(model, input_size, device)
        
        results.append({
            'model_name': model_name,
            'flops': flops_info['flops'],
            'params': flops_info['params'],
            'flops_per_param': flops_info['flops_per_param'],
            'theoretical_throughput_gpu': flops_info['theoretical_throughput_gpu'],
            'theoretical_throughput_cpu': flops_info['theoretical_throughput_cpu'],
            'actual_throughput': flops_info['actual_throughput'],
            'efficiency_ratio': flops_info['efficiency_ratio']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_flops_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Create visualizations
    
    # 1. FLOPs comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['flops'] / 1e9)  # Convert to GFLOPs
    plt.xlabel('Model')
    plt.ylabel('GFLOPs')
    plt.title('Model Computational Complexity (GFLOPs)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(df['flops']):
        plt.text(i, v / 1e9 * 1.01, f'{v / 1e9:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_flops_comparison")
    flops_png, flops_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 2. Throughput comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['theoretical_throughput_gpu'], width, label='Theoretical (GPU)')
    plt.bar(x + width/2, df['actual_throughput'], width, label='Actual')
    
    plt.xlabel('Model')
    plt.ylabel('Throughput (images/second)')
    plt.title('Theoretical vs. Actual Throughput')
    plt.xticks(x, df['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_throughput_comparison")
    throughput_png, throughput_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # 3. Efficiency ratio
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['efficiency_ratio'] * 100)  # Convert to percentage
    plt.xlabel('Model')
    plt.ylabel('Efficiency Ratio (%)')
    plt.title('Model Efficiency Ratio (Actual/Theoretical Throughput)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(df['efficiency_ratio']):
        plt.text(i, v * 100 * 1.01, f'{v * 100:.2f}%', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_efficiency_ratio")
    efficiency_png, efficiency_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return df, csv_path, flops_png


def analyze_layer_distribution(
    model: nn.Module,
    model_name: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Analyze the distribution of FLOPs and parameters across layers.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        input_size: Input tensor size (batch_size, channels, height, width)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with layer-wise analysis, path to CSV file, and path to plot
    """
    try:
        # Calculate layer-wise FLOPs
        df = calculate_layer_flops(model, input_size)
        
        # Check if we got a valid DataFrame
        if df is None or len(df) == 0:
            print(f"No layer-wise FLOPs data available for {model_name}")
            return pd.DataFrame(), "", ""
        
        # Check if we only have a single "total" entry
        if len(df) == 1 and df['layer_name'].iloc[0] == 'total':
            print(f"  ℹ️ Using simplified representation for {model_name} - detailed layer breakdown not available")
            
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{model_name}_layer_flops.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        
        # 1. Top 10 layers by FLOPs
        top_n = min(10, len(df))
        top_layers = df.head(top_n)
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_layers['layer_name'], top_layers['flops_percentage'])
        plt.xlabel('Layer')
        plt.ylabel('FLOPs (%)')
        
        # Use appropriate title based on whether we have detailed layers or just total
        if len(df) == 1 and df['layer_name'].iloc[0] == 'total':
            plt.title(f'Total Computational Cost - {model_name}')
        else:
            plt.title(f'Top {top_n} Layers by Computational Cost - {model_name}')
            
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(top_layers['flops_percentage']):
            plt.text(i, v * 1.01, f'{v:.2f}%', ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{model_name}_layer_distribution")
        save_figure(plt, plot_path, formats=['png', 'svg'])
        
        return df, csv_path, plot_path
    except Exception as e:
        print(f"Error in analyze_layer_distribution: {e}")
        return pd.DataFrame(), "", ""


def calculate_theoretical_memory(
    model: nn.Module,
    batch_size: int = 1,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'float32'
) -> Dict[str, float]:
    """
    Calculate theoretical memory usage for a PyTorch model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        input_size: Input tensor size (channels, height, width)
        precision: Precision of model parameters ('float32', 'float16', or 'int8')
        
    Returns:
        Dictionary with memory usage information in MB
    """
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    
    # Calculate parameter memory
    bytes_per_param = 4 if precision == 'float32' else 2 if precision == 'float16' else 1
    param_memory = params * bytes_per_param / (1024 * 1024)  # Convert to MB
    
    # Calculate input memory
    input_elements = batch_size * np.prod(input_size)
    input_memory = input_elements * bytes_per_param / (1024 * 1024)  # Convert to MB
    
    # Estimate activation memory (rough approximation)
    # Assuming activations are about 4x the parameter size for typical CNNs
    activation_memory = param_memory * 4 * batch_size
    
    # Estimate gradient memory (same as parameter memory during training)
    gradient_memory = param_memory if model.training else 0
    
    # Estimate optimizer state memory (depends on optimizer, using Adam as reference)
    # Adam uses 2 additional states per parameter
    optimizer_memory = param_memory * 2 if model.training else 0
    
    # Total memory
    total_memory = param_memory + input_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'parameter_memory_mb': param_memory,
        'input_memory_mb': input_memory,
        'activation_memory_mb': activation_memory,
        'gradient_memory_mb': gradient_memory,
        'optimizer_memory_mb': optimizer_memory,
        'total_memory_mb': total_memory
    }


if __name__ == "__main__":
    # Example usage
    print("FLOPs analysis module loaded successfully.")
    print("Use calculate_flops() to compute FLOPs for a model.")
    print("Use compare_model_flops() to compare computational complexity across models.")
    print("Use analyze_layer_distribution() to analyze FLOPs distribution across layers.") 