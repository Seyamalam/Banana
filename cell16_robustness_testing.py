import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from cell5_visualization import save_figure
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RobustnessTest:
    """Class to test model robustness against various perturbations."""
    
    def __init__(
        self, 
        model: nn.Module,
        model_name: str,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'models'
    ):
        """
        Initialize the robustness test.
        
        Args:
            model: Model to test
            model_name: Name of the model
            test_loader: DataLoader for test data
            device: Device to run on ('cuda' or 'cpu')
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Store results
        self.results = {}
        
    def _evaluate_with_transform(
        self, 
        transform_fn: Callable,
        perturbation_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model with a specific transform applied to inputs.
        
        Args:
            transform_fn: Function to transform input images
            perturbation_name: Name of the perturbation
            
        Returns:
            Dictionary with evaluation results
        """
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Apply transform to inputs
                transformed_inputs = torch.stack([transform_fn(img) for img in inputs])
                
                # Move to device
                transformed_inputs = transformed_inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(transformed_inputs)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'perturbation': perturbation_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': all_targets,
            'y_pred': all_preds
        }
    
    def test_gaussian_noise(self, std_levels: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5]) -> pd.DataFrame:
        """
        Test robustness against Gaussian noise.
        
        Args:
            std_levels: List of standard deviation levels for noise
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without noise
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different noise levels
        for std in std_levels:
            def add_gaussian_noise(img):
                # Convert to numpy array
                img_np = img.cpu().numpy()
                
                # Add noise
                noise = np.random.normal(0, std, img_np.shape)
                noisy_img = img_np + noise
                
                # Clip to valid range
                noisy_img = np.clip(noisy_img, 0, 1)
                
                # Convert back to tensor
                return torch.from_numpy(noisy_img).float()
            
            result = self._evaluate_with_transform(
                add_gaussian_noise, f'Gaussian Noise (std={std})'
            )
            results.append(result)
        
        # Store results
        self.results['gaussian_noise'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_gaussian_noise_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'gaussian_noise', 'Gaussian Noise Standard Deviation',
            x_values=[0] + std_levels
        )
        
        return df
    
    def test_blur(self, kernel_sizes: List[int] = [3, 5, 7, 9, 11]) -> pd.DataFrame:
        """
        Test robustness against Gaussian blur.
        
        Args:
            kernel_sizes: List of kernel sizes for blur
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without blur
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different blur levels
        for kernel_size in kernel_sizes:
            def apply_blur(img):
                # Convert to PIL image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Apply blur
                blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
                
                # Convert back to tensor
                blurred_np = np.array(blurred_img).astype(np.float32) / 255.0
                blurred_tensor = torch.from_numpy(blurred_np.transpose(2, 0, 1)).float()
                
                return blurred_tensor
            
            result = self._evaluate_with_transform(
                apply_blur, f'Gaussian Blur (kernel={kernel_size})'
            )
            results.append(result)
        
        # Store results
        self.results['blur'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_blur_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'blur', 'Gaussian Blur Kernel Size',
            x_values=[0] + kernel_sizes
        )
        
        return df
    
    def test_brightness(self, factors: List[float] = [0.5, 0.75, 1.25, 1.5, 2.0]) -> pd.DataFrame:
        """
        Test robustness against brightness changes.
        
        Args:
            factors: List of brightness factors (1.0 is original brightness)
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without brightness change
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different brightness levels
        for factor in factors:
            def adjust_brightness(img):
                # Convert to PIL image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Adjust brightness
                enhancer = ImageEnhance.Brightness(img_pil)
                brightened_img = enhancer.enhance(factor)
                
                # Convert back to tensor
                brightened_np = np.array(brightened_img).astype(np.float32) / 255.0
                brightened_tensor = torch.from_numpy(brightened_np.transpose(2, 0, 1)).float()
                
                return brightened_tensor
            
            result = self._evaluate_with_transform(
                adjust_brightness, f'Brightness (factor={factor})'
            )
            results.append(result)
        
        # Store results
        self.results['brightness'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_brightness_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'brightness', 'Brightness Factor',
            x_values=[1.0] + factors
        )
        
        return df
    
    def test_contrast(self, factors: List[float] = [0.5, 0.75, 1.25, 1.5, 2.0]) -> pd.DataFrame:
        """
        Test robustness against contrast changes.
        
        Args:
            factors: List of contrast factors (1.0 is original contrast)
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without contrast change
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different contrast levels
        for factor in factors:
            def adjust_contrast(img):
                # Convert to PIL image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Adjust contrast
                enhancer = ImageEnhance.Contrast(img_pil)
                contrasted_img = enhancer.enhance(factor)
                
                # Convert back to tensor
                contrasted_np = np.array(contrasted_img).astype(np.float32) / 255.0
                contrasted_tensor = torch.from_numpy(contrasted_np.transpose(2, 0, 1)).float()
                
                return contrasted_tensor
            
            result = self._evaluate_with_transform(
                adjust_contrast, f'Contrast (factor={factor})'
            )
            results.append(result)
        
        # Store results
        self.results['contrast'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_contrast_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'contrast', 'Contrast Factor',
            x_values=[1.0] + factors
        )
        
        return df
    
    def test_rotation(self, angles: List[int] = [5, 10, 15, 30, 45, 90]) -> pd.DataFrame:
        """
        Test robustness against rotation.
        
        Args:
            angles: List of rotation angles in degrees
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without rotation
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different rotation angles
        for angle in angles:
            def apply_rotation(img):
                # Convert to PIL image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Apply rotation
                rotated_img = img_pil.rotate(angle)
                
                # Convert back to tensor
                rotated_np = np.array(rotated_img).astype(np.float32) / 255.0
                rotated_tensor = torch.from_numpy(rotated_np.transpose(2, 0, 1)).float()
                
                return rotated_tensor
            
            result = self._evaluate_with_transform(
                apply_rotation, f'Rotation (angle={angle}°)'
            )
            results.append(result)
        
        # Store results
        self.results['rotation'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_rotation_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'rotation', 'Rotation Angle (degrees)',
            x_values=[0] + angles
        )
        
        return df
    
    def test_occlusion(self, sizes: List[int] = [10, 20, 30, 40, 50]) -> pd.DataFrame:
        """
        Test robustness against occlusion (random black patches).
        
        Args:
            sizes: List of occlusion patch sizes in pixels
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without occlusion
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different occlusion sizes
        for size in sizes:
            def apply_occlusion(img):
                # Convert to numpy array
                img_np = img.cpu().numpy()
                
                # Get image dimensions
                _, h, w = img_np.shape
                
                # Generate random position for occlusion
                x = random.randint(0, w - size)
                y = random.randint(0, h - size)
                
                # Apply occlusion
                img_occluded = img_np.copy()
                img_occluded[:, y:y+size, x:x+size] = 0
                
                # Convert back to tensor
                return torch.from_numpy(img_occluded).float()
            
            result = self._evaluate_with_transform(
                apply_occlusion, f'Occlusion (size={size}px)'
            )
            results.append(result)
        
        # Store results
        self.results['occlusion'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_occlusion_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'occlusion', 'Occlusion Patch Size (pixels)',
            x_values=[0] + sizes
        )
        
        return df
    
    def test_jpeg_compression(self, qualities: List[int] = [90, 80, 70, 60, 50, 40, 30, 20, 10]) -> pd.DataFrame:
        """
        Test robustness against JPEG compression.
        
        Args:
            qualities: List of JPEG quality levels (100 is best quality)
            
        Returns:
            DataFrame with results
        """
        results = []
        
        # First, evaluate without compression
        identity_transform = lambda x: x
        baseline_result = self._evaluate_with_transform(identity_transform, 'No Perturbation')
        results.append(baseline_result)
        
        # Then, evaluate with different compression levels
        for quality in qualities:
            def apply_jpeg_compression(img):
                # Convert to PIL image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Save with JPEG compression
                import io
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                
                # Load compressed image
                compressed_img = Image.open(buffer)
                
                # Convert back to tensor
                compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
                compressed_tensor = torch.from_numpy(compressed_np.transpose(2, 0, 1)).float()
                
                return compressed_tensor
            
            result = self._evaluate_with_transform(
                apply_jpeg_compression, f'JPEG Compression (quality={quality})'
            )
            results.append(result)
        
        # Store results
        self.results['jpeg_compression'] = results
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'perturbation': r['perturbation'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in results
        ])
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_jpeg_compression_robustness.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        self._plot_metric_vs_perturbation(
            df, 'jpeg_compression', 'JPEG Quality',
            x_values=[100] + qualities
        )
        
        return df
    
    def _plot_metric_vs_perturbation(
        self, 
        df: pd.DataFrame,
        perturbation_type: str,
        x_label: str,
        x_values: List[float] = None
    ) -> None:
        """
        Plot metrics vs perturbation level.
        
        Args:
            df: DataFrame with results
            perturbation_type: Type of perturbation
            x_label: Label for x-axis
            x_values: Values for x-axis (if None, use indices)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            plt.plot(x_values if x_values else range(len(df)), df[metric], marker='o', label=metric.capitalize())
        
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel('Score')
        plt.title(f'Model Robustness to {perturbation_type.replace("_", " ").title()}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_{perturbation_type}_robustness")
        save_figure(plt, base_filename, formats=['png', 'svg'])
    
    def run_all_tests(self) -> Dict[str, pd.DataFrame]:
        """
        Run all robustness tests.
        
        Returns:
            Dictionary mapping test names to result DataFrames
        """
        results = {}
        
        print("Testing robustness to Gaussian noise...")
        results['gaussian_noise'] = self.test_gaussian_noise()
        
        print("Testing robustness to blur...")
        results['blur'] = self.test_blur()
        
        print("Testing robustness to brightness changes...")
        results['brightness'] = self.test_brightness()
        
        print("Testing robustness to contrast changes...")
        results['contrast'] = self.test_contrast()
        
        print("Testing robustness to rotation...")
        results['rotation'] = self.test_rotation()
        
        print("Testing robustness to occlusion...")
        results['occlusion'] = self.test_occlusion()
        
        print("Testing robustness to JPEG compression...")
        results['jpeg_compression'] = self.test_jpeg_compression()
        
        # Create summary visualization
        self._create_summary_visualization()
        
        return results
    
    def _create_summary_visualization(self) -> None:
        """Create summary visualization of all robustness tests."""
        # Extract baseline and worst-case accuracies for each test
        summary_data = []
        
        for test_name, results in self.results.items():
            baseline_acc = results[0]['accuracy']
            worst_acc = min([r['accuracy'] for r in results])
            acc_drop = baseline_acc - worst_acc
            
            summary_data.append({
                'test': test_name.replace('_', ' ').title(),
                'baseline_accuracy': baseline_acc,
                'worst_accuracy': worst_acc,
                'accuracy_drop': acc_drop,
                'relative_drop': acc_drop / baseline_acc if baseline_acc > 0 else 0
            })
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_robustness_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Sort by relative drop
        summary_df = summary_df.sort_values('relative_drop', ascending=False)
        
        # Plot relative accuracy drop
        plt.bar(summary_df['test'], summary_df['relative_drop'] * 100)
        
        # Add labels and title
        plt.xlabel('Perturbation Type')
        plt.ylabel('Relative Accuracy Drop (%)')
        plt.title(f'Model Robustness Summary - {self.model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(summary_df['relative_drop']):
            plt.text(i, v * 100 * 1.01, f'{v * 100:.1f}%', ha='center')
        
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.join(self.output_dir, f"{self.model_name}_robustness_summary")
        save_figure(plt, base_filename, formats=['png', 'svg'])


def compare_model_robustness(
    models: List[Dict[str, Any]],
    test_loader: torch.utils.data.DataLoader,
    perturbation_type: str = 'gaussian_noise',
    output_dir: str = 'models'
) -> Tuple[pd.DataFrame, str, str]:
    """
    Compare robustness of multiple models against a specific perturbation.
    
    Args:
        models: List of dictionaries, each containing:
               - 'name': Model name
               - 'model': PyTorch model
        test_loader: DataLoader for test data
        perturbation_type: Type of perturbation to test
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results, path to CSV file, and path to plot
    """
    # Test each model
    results = []
    
    for model_dict in models:
        model_name = model_dict['name']
        model = model_dict['model']
        
        print(f"Testing robustness of {model_name}...")
        
        # Create robustness test
        robustness_test = RobustnessTest(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            output_dir=output_dir
        )
        
        # Run specific test
        if perturbation_type == 'gaussian_noise':
            df = robustness_test.test_gaussian_noise()
        elif perturbation_type == 'blur':
            df = robustness_test.test_blur()
        elif perturbation_type == 'brightness':
            df = robustness_test.test_brightness()
        elif perturbation_type == 'contrast':
            df = robustness_test.test_contrast()
        elif perturbation_type == 'rotation':
            df = robustness_test.test_rotation()
        elif perturbation_type == 'occlusion':
            df = robustness_test.test_occlusion()
        elif perturbation_type == 'jpeg_compression':
            df = robustness_test.test_jpeg_compression()
        else:
            raise ValueError(f"Unsupported perturbation type: {perturbation_type}")
        
        # Extract baseline and worst-case accuracies
        baseline_acc = df.iloc[0]['accuracy']
        worst_acc = df['accuracy'].min()
        acc_drop = baseline_acc - worst_acc
        
        results.append({
            'model_name': model_name,
            'baseline_accuracy': baseline_acc,
            'worst_accuracy': worst_acc,
            'accuracy_drop': acc_drop,
            'relative_drop': acc_drop / baseline_acc if baseline_acc > 0 else 0
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"model_robustness_comparison_{perturbation_type}.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Sort by relative drop
    comparison_df = comparison_df.sort_values('relative_drop')
    
    # Plot relative accuracy drop
    plt.bar(comparison_df['model_name'], comparison_df['relative_drop'] * 100)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Relative Accuracy Drop (%)')
    plt.title(f'Model Robustness Comparison - {perturbation_type.replace("_", " ").title()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(comparison_df['relative_drop']):
        plt.text(i, v * 100 * 1.01, f'{v * 100:.1f}%', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, f"model_robustness_comparison_{perturbation_type}")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return comparison_df, csv_path, png_path


if __name__ == "__main__":
    # Example usage
    print("Robustness testing module loaded successfully.")
    print("Use RobustnessTest class to test model robustness against various perturbations.")
    print("Use compare_model_robustness() to compare robustness across multiple models.") 