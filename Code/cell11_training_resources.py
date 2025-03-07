import time
import os
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import csv
from cell1_imports_and_constants import CLASS_NAMES
from cell5_visualization import save_figure

class ResourceTracker:
    """Class to track resources during model training and inference."""
    
    def __init__(self, model_name: str, output_dir: str = "models"):
        """
        Initialize the resource tracker.
        
        Args:
            model_name: Name of the model being tracked
            output_dir: Directory to save resource tracking results
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        self.process = psutil.Process(os.getpid())
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def start(self) -> None:
        """Start tracking resources."""
        self.start_time = time.time()
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        
    def update(self) -> None:
        """Update resource tracking metrics."""
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        # Track CPU memory usage (in MB)
        memory_info = self.process.memory_info()
        self.memory_usage.append(memory_info.rss / (1024 * 1024))
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            self.gpu_memory_usage.append(gpu_memory)
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking resources and return metrics.
        
        Returns:
            Dictionary containing resource metrics
        """
        self.end_time = time.time()
        training_time = self.end_time - self.start_time
        
        peak_memory = max(self.memory_usage) if self.memory_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        
        peak_gpu_memory = max(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        avg_gpu_memory = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        
        metrics = {
            "model_name": self.model_name,
            "training_time_seconds": training_time,
            "peak_memory_mb": peak_memory,
            "avg_memory_mb": avg_memory,
            "peak_gpu_memory_mb": peak_gpu_memory,
            "avg_gpu_memory_mb": avg_gpu_memory
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Save resource metrics to CSV file.
        
        Args:
            metrics: Dictionary of resource metrics
            
        Returns:
            Path to the saved CSV file
        """
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_resource_metrics.csv")
        
        # Convert metrics to rows for CSV
        rows = [[key, value] for key, value in metrics.items()]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerows(rows)
            
        return csv_path
    
    def plot_memory_usage(self) -> Tuple[str, str]:
        """
        Plot memory usage over time and save as PNG and SVG.
        
        Returns:
            Tuple of paths to saved PNG and SVG files, or (None, None) if files weren't saved
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.memory_usage, label='CPU Memory (MB)')
        
        if torch.cuda.is_available() and self.gpu_memory_usage:
            plt.plot(self.timestamps, self.gpu_memory_usage, label='GPU Memory (MB)')
            
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage During Training - {self.model_name}')
        plt.legend()
        plt.grid(True)
        
        # Save figure in multiple formats
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_memory_usage")
        paths = save_figure(plt, base_filename, formats=['png', 'svg'])
        
        # If save_figure returns None, provide default return values
        if paths is None:
            return None, None
        
        return paths


def measure_training_resources(model_name: str, train_function, *args, **kwargs) -> Dict[str, Any]:
    """
    Measure resources used during model training.
    
    Args:
        model_name: Name of the model
        train_function: Function that trains the model
        *args, **kwargs: Arguments to pass to the train function
        
    Returns:
        Dictionary of resource metrics
    """
    tracker = ResourceTracker(model_name)
    tracker.start()
    
    # Create a simple wrapper that updates resources periodically during training
    # without requiring changes to the original training function
    import threading
    
    # Flag to control the resource tracking thread
    tracking_active = True
    
    # Create a thread that updates resource tracking every second
    def resource_tracking_thread():
        while tracking_active:
            tracker.update()
            time.sleep(1.0)  # Update every second
    
    # Start the resource tracking thread
    tracking_thread = threading.Thread(target=resource_tracking_thread)
    tracking_thread.daemon = True
    tracking_thread.start()
    
    try:
        # Run the training function with original arguments
        result = train_function(*args, **kwargs)
    finally:
        # Stop the resource tracking thread
        tracking_active = False
        tracking_thread.join(timeout=2.0)  # Wait for the thread to finish
    
    # Stop tracking and get metrics
    metrics = tracker.stop()
    
    # Save metrics to CSV
    csv_path = tracker.save_metrics(metrics)
    print(f"Resource metrics saved to {csv_path}")
    
    # Plot memory usage
    try:
        png_path, svg_path = tracker.plot_memory_usage()
        if png_path and svg_path:
            print(f"Memory usage plot saved to {png_path} and {svg_path}")
        else:
            print("Memory usage plot could not be saved")
    except Exception as e:
        print(f"Error plotting memory usage: {e}")
    
    return metrics


def compare_training_resources(model_metrics: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Compare training resources across multiple models and create visualizations.
    
    Args:
        model_metrics: List of dictionaries containing resource metrics for different models
        
    Returns:
        Tuple of paths to saved comparison CSV and plot files
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(model_metrics)
    
    # Save comparison to CSV
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_resource_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Create bar charts for comparison
    metrics_to_plot = [
        'training_time_seconds', 
        'peak_memory_mb', 
        'avg_memory_mb'
    ]
    
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        metrics_to_plot.extend(['peak_gpu_memory_mb', 'avg_gpu_memory_mb'])
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot)))
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i] if len(metrics_to_plot) > 1 else axes
        ax.bar(df['model_name'], df[metric])
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.split('_')[-1].upper())
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for j, v in enumerate(df[metric]):
            ax.text(j, v * 1.01, f'{v:.2f}', ha='center')
            
        # Rotate x-axis labels if there are many models
        if len(df) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure in multiple formats
    base_filename = os.path.join(output_dir, "model_resource_comparison")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return csv_path, png_path


def estimate_energy_consumption(training_time: float, gpu_used: bool = False) -> float:
    """
    Estimate energy consumption based on training time and hardware used.
    This is a simplified estimation and should be replaced with actual measurements
    when possible.
    
    Args:
        training_time: Training time in seconds
        gpu_used: Whether GPU was used for training
        
    Returns:
        Estimated energy consumption in watt-hours
    """
    # Rough estimates of power consumption
    # CPU-only: ~100W, GPU: additional ~250W
    base_power = 100  # watts
    gpu_power = 250 if gpu_used else 0  # watts
    
    total_power = base_power + gpu_power
    energy_wh = (total_power * training_time) / 3600  # convert to watt-hours
    
    return energy_wh


def calculate_carbon_footprint(energy_wh: float, carbon_intensity: float = 475) -> float:
    """
    Calculate carbon footprint based on energy consumption.
    
    Args:
        energy_wh: Energy consumption in watt-hours
        carbon_intensity: Carbon intensity in gCO2/kWh (default: 475, global average)
        
    Returns:
        Carbon footprint in grams of CO2
    """
    # Convert Wh to kWh and multiply by carbon intensity
    carbon_g = (energy_wh / 1000) * carbon_intensity
    
    return carbon_g


if __name__ == "__main__":
    # Example usage
    print("Resource tracking module loaded successfully.")
    print("Use measure_training_resources() to track resources during model training.")
    print("Use compare_training_resources() to compare resources across multiple models.") 