import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil

# Import modules
from cell1_imports_and_constants import set_seed, NUM_CLASSES
from cell2_dataset import load_data
from cell3_model import BananaLeafCNN
from cell4_training import train, validate
from cell6_utils import save_checkpoint, load_checkpoint, evaluate_model
from cell11_training_resources import measure_training_resources, compare_training_resources, estimate_energy_consumption, calculate_carbon_footprint
from cell12_statistical_testing import statistical_significance_test, mcnemar_test
from cell13_efficiency_metrics import calculate_advanced_efficiency_metrics, calculate_pareto_frontier, calculate_model_size
from cell14_ablation_studies import AblationStudy, create_model_variant, change_dropout_rate, change_activation, remove_layer, change_normalization
from cell15_flops_analysis import calculate_flops, compare_model_flops, analyze_layer_distribution, calculate_theoretical_memory
from cell16_robustness_testing import RobustnessTest, compare_model_robustness
from cell18_deployment_metrics import benchmark_deployment_metrics, compare_deployment_metrics

# Import model zoo for comparison
from cell8_model_zoo import (
    build_mobilenet_v2, 
    build_efficientnet_b0, 
    build_resnet18, 
    build_shufflenet_v2,
    get_available_classification_models,
    create_model_adapter,
    ModelInfo,
    get_model_info_from_model,
    load_pretrained_models
)

from cell5_visualization import (
    plot_confusion_matrix, 
    visualize_predictions, 
    save_classification_report, 
    plot_roc_curves, 
    plot_precision_recall_curves,
    visualize_model_architecture,
    plot_training_metrics,
    plot_class_distribution,
    save_figure
)


def parse_args():
    parser = argparse.ArgumentParser(description='Banana Leaf Disease Classification Analysis')
    
    # Basic arguments
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Analysis options
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--robustness', action='store_true', help='Run robustness tests')
    parser.add_argument('--deployment', action='store_true', help='Run deployment metrics')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--load_all_models', action='store_true', help='Load all available models for comparison')
    
    # Model selection
    available_models = ['banana_leaf_cnn'] + get_available_classification_models() + ['mobilenet_v2', 'efficientnet_b0', 'resnet18', 'shufflenet_v2']
    # Remove duplicates while preserving order
    available_models = list(dict.fromkeys(available_models))
    
    parser.add_argument('--models', nargs='+', default=['banana_leaf_cnn'], 
                        choices=available_models,
                        help='Models to analyze')
    
    args = parser.parse_args()
    
    # If no analysis options are specified, run all analyses
    if not (args.train or args.evaluate or args.ablation or args.robustness or args.deployment or args.all):
        print("No analysis options specified. Running all analyses with default model.")
        args.all = True
    
    return args


def train_model(model, model_name, train_loader, val_loader, args):
    """Train a model and save checkpoints."""
    print(f"Training {model_name}...")
    
    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create training metrics directory
    metrics_dir = os.path.join(model_dir, 'training')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot class distribution before training
    try:
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        dist_path = os.path.join(metrics_dir, f"{model_name}_class_distribution")
        plot_class_distribution(all_labels, val_loader, save_path=dist_path, formats=['png', 'svg'])
    except Exception as e:
        print(f"Warning: Could not plot class distribution: {e}")
    
    # Training loop with resource tracking
    def training_function(model, train_loader, val_loader, criterion, optimizer, device, epochs):
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_report = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch: {epoch+1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, 
                               os.path.join(args.output_dir, f"{model_name}_best.pt"))
        
        # Save final model
        save_checkpoint(model, optimizer, epochs-1, val_acc, 
                       os.path.join(args.output_dir, f"{model_name}_final.pt"))
        
        # Plot training metrics
        metrics_path = os.path.join(metrics_dir, f"{model_name}_training_metrics")
        plot_training_metrics(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            save_path=metrics_path,
            formats=['png', 'svg']
        )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    # Measure resources during training
    training_metrics = measure_training_resources(
        model_name,
        training_function,
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs
    )
    
    return training_metrics


def compare_confusion_matrices(results, output_dir):
    """
    Create a side-by-side comparison of confusion matrices for all models.
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save comparison visualizations
    """
    print("Generating confusion matrix comparison...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    from cell1_imports_and_constants import IDX_TO_CLASS
    class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    
    # Determine grid size for the plot
    num_models = len(results)
    ncols = min(3, num_models)  # Maximum 3 columns
    nrows = (num_models + ncols - 1) // ncols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    
    # Plot each confusion matrix
    for i, result in enumerate(results):
        model_name = result['name']
        cm = result['confusion_matrix']
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Add subplot
        ax = fig.add_subplot(nrows, ncols, i + 1)
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix: {model_name}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    comparison_path = os.path.join(output_dir, "confusion_matrix_comparison")
    png_path, svg_path = save_figure(plt, comparison_path, formats=['png', 'svg'])
    
    print(f"Confusion matrix comparison saved to {png_path}")
    
    # Return path to saved figure
    return png_path


def compare_classification_metrics(results, output_dir):
    """
    Create visualizations comparing classification metrics (precision, recall, F1) across models.
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save comparison visualizations
    """
    print("Generating classification metrics comparison...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    from cell1_imports_and_constants import IDX_TO_CLASS
    class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    num_classes = len(class_names)
    
    # Prepare data for each model
    model_names = [result['name'] for result in results]
    
    # Calculate per-class precision, recall, and F1 for each model
    from sklearn.metrics import precision_recall_fscore_support
    
    metrics_data = []
    
    for result in results:
        model_name = result['name']
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        # Skip if no predictions
        if len(y_true) == 0 or len(y_pred) == 0:
            continue
        
        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(num_classes), zero_division=0
        )
        
        # Store results
        for i in range(num_classes):
            metrics_data.append({
                'Model': model_name,
                'Class': class_names[i],
                'Precision': precision[i],
                'Recall': recall[i],
                'F1 Score': f1[i],
                'Support': support[i]
            })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, "classification_metrics_by_class.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Create comparison visualizations for each metric
    metrics_to_plot = ['Precision', 'Recall', 'F1 Score']
    all_paths = {}
    
    for metric in metrics_to_plot:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Reshape data for bar chart
        pivot_df = metrics_df.pivot(index='Class', columns='Model', values=metric)
        
        # Create bar chart
        ax = pivot_df.plot(kind='bar')
        
        plt.xlabel('Class')
        plt.ylabel(metric)
        plt.title(f'{metric} by Class Across Models')
        plt.legend(title='Model')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        metric_path = os.path.join(output_dir, f"comparison_{metric.lower().replace(' ', '_')}")
        png_path, svg_path = save_figure(plt, metric_path, formats=['png', 'svg'])
        all_paths[metric] = png_path
    
    # Create heatmap showing F1 score for each model and class
    plt.figure(figsize=(12, 8))
    pivot_df = metrics_df.pivot(index='Class', columns='Model', values='F1 Score')
    sns.heatmap(pivot_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title('F1 Score by Class and Model')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, "f1_score_heatmap")
    png_path, svg_path = save_figure(plt, heatmap_path, formats=['png', 'svg'])
    all_paths['F1 Heatmap'] = png_path
    
    # Create a spider plot for each class showing all metrics across models
    for class_name in class_names:
        class_df = metrics_df[metrics_df['Class'] == class_name]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Metrics to include in the spider plot
        radar_metrics = ['Precision', 'Recall', 'F1 Score']
        
        # Number of variables
        N = len(radar_metrics)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the chart for each model
        for model_name in model_names:
            model_data = class_df[class_df['Model'] == model_name]
            if len(model_data) == 0:
                continue
                
            values = model_data[radar_metrics].values[0].tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        
        # Set limits for consistent visualization
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'Metrics for Class: {class_name}')
        
        # Save figure
        class_path = os.path.join(output_dir, f"metrics_radar_{class_name.replace(' ', '_')}")
        png_path, svg_path = save_figure(plt, class_path, formats=['png', 'svg'])
    
    print(f"Classification metrics comparison saved to {output_dir}")
    
    # Return path to main metrics visualization
    return csv_path


def evaluate_models(models, model_names, test_loader, args):
    """Evaluate models and compare their performance."""
    print("Evaluating models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    for model, model_name in zip(models, model_names):
        print(f"Evaluating {model_name}...")
        model = model.to(device)
        model.eval()
        
        # Evaluate model
        metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
        
        # Store results
        results.append({
            'name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'y_true': y_true,
            'y_pred': y_pred,
            'confusion_matrix': metrics['confusion_matrix']
        })
        
        # Create evaluation output directory
        eval_dir = os.path.join(args.output_dir, 'evaluation', model_name)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Plot and save confusion matrix
        cm_path = os.path.join(eval_dir, f"{model_name}_confusion_matrix")
        plot_confusion_matrix(metrics['confusion_matrix'], cm_path, formats=['png', 'svg'])
        
        # Generate classification report
        from sklearn.metrics import classification_report
        from cell1_imports_and_constants import IDX_TO_CLASS
        report = classification_report(y_true, y_pred, target_names=[IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))], zero_division=0)
        
        # Save classification report
        report_path = os.path.join(eval_dir, f"{model_name}_classification_report.csv")
        save_classification_report(report, report_path)
        
        # Visualize sample predictions
        try:
            # Get a batch of test data
            test_iterator = iter(test_loader)
            images, labels = next(test_iterator)
            images = images.to(device)
            labels = labels.to(device)
            
            # Make predictions
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            
            # Create a grid of images with predictions
            fig = plt.figure(figsize=(12, 8))
            for i in range(min(16, len(images))):
                ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
                # Move tensor to CPU and convert to numpy for plotting
                img = images[i].cpu().permute(1, 2, 0).numpy()
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                # Color based on whether prediction is correct
                title_color = 'green' if preds[i] == labels[i] else 'red'
                true_label = IDX_TO_CLASS[labels[i].item()]
                pred_label = IDX_TO_CLASS[preds[i].item()]
                ax.set_title(f"T: {true_label}\nP: {pred_label}", color=title_color)
            
            plt.tight_layout()
            
            # Save the figure
            vis_path = os.path.join(eval_dir, f"{model_name}_predictions")
            save_figure(plt, vis_path, formats=['png', 'svg'])
        except Exception as e:
            print(f"Warning: Could not visualize predictions: {e}")
            
        # Generate ROC curve
        try:
            roc_path = os.path.join(eval_dir, f"{model_name}_roc_curve")
            plot_roc_curves(model, test_loader, device, save_path=roc_path, formats=['png', 'svg'])
            
            # Generate precision-recall curve
            pr_path = os.path.join(eval_dir, f"{model_name}_precision_recall_curve")
            plot_precision_recall_curves(model, test_loader, device, save_path=pr_path, formats=['png', 'svg'])
        except Exception as e:
            print(f"Warning: Could not generate ROC or PR curves: {e}")
        
        # Visualize model architecture
        try:
            arch_path = os.path.join(eval_dir, f"{model_name}_architecture")
            visualize_model_architecture(model, save_path=arch_path, formats=['png', 'svg'])
        except Exception as e:
            print(f"Warning: Could not visualize model architecture: {e}")
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"Evaluation results saved to {eval_dir}")
    
    # If we have multiple models, generate comparison visualizations
    if len(results) > 1:
        # Create evaluation comparisons directory
        comparison_dir = os.path.join(args.output_dir, 'comparisons', 'evaluation')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Generate confusion matrix comparison
        compare_confusion_matrices(results, comparison_dir)
        
        # Generate classification metrics comparison
        compare_classification_metrics(results, comparison_dir)
        
        # Statistical significance testing and save to comparison directory
        print("\nPerforming statistical significance testing...")
        p_value_df, csv_path, plot_path = statistical_significance_test(
            results, 
            output_dir=comparison_dir
        )
        print(f"Statistical significance results saved to {csv_path}")
        
        # McNemar test and save to comparison directory
        mcnemar_df, mcnemar_csv = mcnemar_test(
            results, 
            output_dir=comparison_dir
        )
        print(f"McNemar test results saved to {mcnemar_csv}")
        
        # Create tabular comparison of overall metrics
        overall_metrics = []
        for result in results:
            overall_metrics.append({
                'Model': result['name'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1']
            })
        
        # Save as CSV
        overall_df = pd.DataFrame(overall_metrics)
        overall_csv = os.path.join(comparison_dir, "overall_metrics_comparison.csv")
        overall_df.to_csv(overall_csv, index=False)
        
        # Create bar chart for each metric
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            plt.figure(figsize=(10, 6))
            plt.bar(overall_df['Model'], overall_df[metric])
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.title(f'Comparison of {metric} Across Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            metric_path = os.path.join(comparison_dir, f"overall_{metric.lower().replace(' ', '_')}")
            save_figure(plt, metric_path, formats=['png', 'svg'])
        
        print(f"Overall metrics comparison saved to {overall_csv}")
    else:
        # If only one model, still save the results to the comparisons directory for consistency
        if len(results) == 1:
            comparison_dir = os.path.join(args.output_dir, 'comparisons', 'evaluation')
            os.makedirs(comparison_dir, exist_ok=True)
            
            overall_metrics = [{
                'Model': results[0]['name'],
                'Accuracy': results[0]['accuracy'],
                'Precision': results[0]['precision'],
                'Recall': results[0]['recall'],
                'F1 Score': results[0]['f1']
            }]
            
            overall_df = pd.DataFrame(overall_metrics)
            overall_csv = os.path.join(comparison_dir, "overall_metrics.csv")
            overall_df.to_csv(overall_csv, index=False)
    
    return results


def run_robustness_tests(model, model_name, test_loader, args):
    """Run robustness tests on model."""
    print(f"Running robustness tests for {model_name}...")
    
    # Check disk space before running
    if not check_disk_space(min_space_mb=200):
        print(f"⚠️ Warning: Skipping robustness tests for {model_name} due to low disk space.")
        return {'baseline': {'accuracy': 0}, 'error': 'Low disk space'}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create robustness test
        robustness_test = RobustnessTest(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            device=device,
            output_dir=os.path.join(args.output_dir, 'robustness')
        )
        
        # Run tests
        results = robustness_test.run_all_tests()
        print(f"Robustness test results saved to {os.path.join(args.output_dir, 'robustness')}")
        
        return results
    except Exception as e:
        print(f"❌ Error running robustness tests for {model_name}: {e}")
        return {'baseline': {'accuracy': 0}, 'error': str(e)}


def run_deployment_metrics(model, model_name, args):
    """Run deployment metrics analysis."""
    print(f"Running deployment metrics analysis for {model_name}...")
    
    # Check disk space before running
    if not check_disk_space(min_space_mb=200):
        print(f"⚠️ Warning: Skipping deployment metrics for {model_name} due to low disk space.")
        return None
    
    try:
        # Benchmark deployment metrics
        summary_df, csv_path, plot_path = benchmark_deployment_metrics(
            model=model,
            model_name=model_name,
            input_size=(1, 3, 224, 224),
            batch_sizes=[1, 4, 8, 16, 32],
            output_dir=os.path.join(args.output_dir, 'deployment')
        )
        
        print(f"Deployment metrics saved to {csv_path}")
        return summary_df
    except Exception as e:
        print(f"❌ Error analyzing deployment metrics for {model_name}: {e}")
        # Create a minimal DataFrame with basic info
        return pd.DataFrame({
            'model_name': [model_name],
            'error': [str(e)]
        })


def run_efficiency_metrics(models, model_names, evaluation_results, args):
    """Run efficiency metrics analysis."""
    print("Running efficiency metrics analysis...")
    
    # Check disk space before running
    if not check_disk_space(min_space_mb=300):
        print("⚠️ Warning: Skipping efficiency metrics due to low disk space.")
        return None
    
    try:
        # Prepare data for efficiency metrics
        model_results = []
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            # Get model size
            model_size = calculate_model_size(model)
            
            # Get inference time (placeholder - should be measured properly)
            inference_time = 0.01  # seconds per image
            
            # Get training time (placeholder - should be from actual training)
            training_time = 300  # seconds
            
            model_results.append({
                'name': model_name,
                'model': model,
                'accuracy': evaluation_results[i]['accuracy'],
                'inference_time': inference_time,
                'training_time': training_time,
                'model_size': model_size
            })
        
        # Calculate advanced efficiency metrics
        metrics_df, csv_path, plot_path = calculate_advanced_efficiency_metrics(
            model_results, 
            output_dir=os.path.join(args.output_dir, 'efficiency')
        )
        
        print(f"Efficiency metrics saved to {csv_path}")
        
        # Calculate Pareto frontier
        try:
            pareto_df, pareto_csv, pareto_plot = calculate_pareto_frontier(
                model_results,
                output_dir=os.path.join(args.output_dir, 'efficiency')
            )
            print(f"Pareto frontier analysis saved to {pareto_csv}")
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate Pareto frontier: {e}")
        
        # Compare FLOPs
        try:
            if check_disk_space(min_space_mb=100):
                flops_df, flops_csv, flops_plot = compare_model_flops(
                    model_results,
                    output_dir=os.path.join(args.output_dir, 'efficiency')
                )
                print(f"FLOPs comparison saved to {flops_csv}")
            else:
                print("⚠️ Warning: Skipping FLOPs comparison due to low disk space")
        except Exception as e:
            print(f"⚠️ Warning: Could not compare FLOPs: {e}")
        
        # NEW: Analyze layer distribution for each model
        if check_disk_space(min_space_mb=200):
            print("\n📊 Analyzing FLOPs distribution across layers for each model...")
            layers_dir = os.path.join(args.output_dir, 'efficiency', 'layer_analysis')
            os.makedirs(layers_dir, exist_ok=True)
            
            for model, model_name in zip(models, model_names):
                try:
                    print(f"  Analyzing layer distribution for {model_name}...")
                    layer_results = analyze_layer_distribution(
                        model,
                        model_name,
                        input_size=(1, 3, 224, 224),
                        output_dir=layers_dir
                    )
                    
                    # Check if we got actual results or just a total FLOPs count
                    if isinstance(layer_results, tuple) and len(layer_results) == 3:
                        layer_df, layer_csv, layer_plot = layer_results
                        print(f"  ✅ Layer distribution for {model_name} saved to {layer_csv}")
                    else:
                        print(f"  ✅ Total FLOPs for {model_name} calculated, but detailed layer analysis not available")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not analyze layer distribution for {model_name}: {e}")
        else:
            print("⚠️ Warning: Skipping layer analysis due to low disk space")
        
        # NEW: Calculate theoretical memory usage for each model
        if check_disk_space(min_space_mb=100):
            print("\n📊 Calculating theoretical memory usage for each model...")
            memory_data = []
            
            for model, model_name in zip(models, model_names):
                try:
                    # Calculate memory for different batch sizes and precisions
                    batch_sizes = [1, 4, 16, 32]
                    memory_info = {}
                    
                    # For float32 (default)
                    memory_info['float32'] = calculate_theoretical_memory(
                        model, batch_size=1, precision='float32'
                    )
                    
                    # For float16 (half precision)
                    memory_info['float16'] = calculate_theoretical_memory(
                        model, batch_size=1, precision='float16'
                    )
                    
                    # For batch size scaling
                    batch_memory = []
                    for batch_size in batch_sizes:
                        mem = calculate_theoretical_memory(
                            model, batch_size=batch_size, precision='float32'
                        )
                        batch_memory.append({
                            'batch_size': batch_size,
                            'total_memory_mb': mem['total_memory_mb']
                        })
                    
                    # Add to data
                    memory_data.append({
                        'model_name': model_name,
                        'params': sum(p.numel() for p in model.parameters()),
                        'param_memory_fp32_mb': memory_info['float32']['parameter_memory_mb'],
                        'param_memory_fp16_mb': memory_info['float16']['parameter_memory_mb'],
                        'activation_memory_mb': memory_info['float32']['activation_memory_mb'],
                        'total_memory_mb': memory_info['float32']['total_memory_mb'],
                        'total_memory_fp16_mb': memory_info['float16']['total_memory_mb'],
                        'batch_memory': batch_memory
                    })
                    
                    print(f"  ✅ Memory calculation for {model_name} completed")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not calculate memory usage for {model_name}: {e}")
            
            # Save memory data to CSV
            if memory_data:
                try:
                    mem_df = pd.DataFrame([{
                        'model_name': data['model_name'],
                        'params': data['params'],
                        'param_memory_fp32_mb': data['param_memory_fp32_mb'],
                        'param_memory_fp16_mb': data['param_memory_fp16_mb'],
                        'activation_memory_mb': data['activation_memory_mb'],
                        'total_memory_mb': data['total_memory_mb'],
                        'total_memory_fp16_mb': data['total_memory_fp16_mb']
                    } for data in memory_data])
                    
                    mem_csv = os.path.join(args.output_dir, 'efficiency', 'memory_usage.csv')
                    mem_df.to_csv(mem_csv, index=False)
                    
                    # Create memory usage comparison chart
                    plt.figure(figsize=(12, 6))
                    x = np.arange(len(mem_df))
                    width = 0.35
                    
                    plt.bar(x - width/2, mem_df['total_memory_mb'], width, label='FP32')
                    plt.bar(x + width/2, mem_df['total_memory_fp16_mb'], width, label='FP16')
                    
                    plt.xlabel('Model')
                    plt.ylabel('Memory Usage (MB)')
                    plt.title('Theoretical Memory Usage by Precision')
                    plt.xticks(x, mem_df['model_name'], rotation=45, ha='right')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # Save figure
                    mem_plot = os.path.join(args.output_dir, 'efficiency', 'memory_usage_comparison')
                    save_figure(plt, mem_plot, formats=['png', 'svg'])
                    
                    # Create batch scaling chart
                    plt.figure(figsize=(12, 6))
                    
                    for data in memory_data:
                        batch_sizes = [item['batch_size'] for item in data['batch_memory']]
                        memory_values = [item['total_memory_mb'] for item in data['batch_memory']]
                        plt.plot(batch_sizes, memory_values, marker='o', label=data['model_name'])
                    
                    plt.xlabel('Batch Size')
                    plt.ylabel('Memory Usage (MB)')
                    plt.title('Memory Scaling with Batch Size')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save figure
                    batch_plot = os.path.join(args.output_dir, 'efficiency', 'batch_memory_scaling')
                    save_figure(plt, batch_plot, formats=['png', 'svg'])
                    
                    print(f"Memory usage analysis saved to {mem_csv}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save memory usage analysis: {e}")
        else:
            print("⚠️ Warning: Skipping memory usage analysis due to low disk space")
        
        return metrics_df
    except Exception as e:
        print(f"❌ Error analyzing efficiency metrics: {e}")
        return None


def save_comprehensive_comparison(models, model_names, evaluation_results, deployment_results=None, 
                                 efficiency_results=None, robustness_results=None, args=None):
    """
    Save a comprehensive comparison of all models including metrics from all analyses.
    
    Args:
        models: List of model objects
        model_names: List of model names
        evaluation_results: List of evaluation result dictionaries
        deployment_results: List of deployment result dataframes
        efficiency_results: Efficiency metrics dataframe
        robustness_results: List of robustness result dictionaries
        args: Arguments
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Create comparison directory
    comparison_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Use ModelInfo to get structured information about each model
    from cell8_model_zoo import ModelInfo, get_model_info_from_model
    
    # Import energy estimation functions
    from cell11_training_resources import estimate_energy_consumption, calculate_carbon_footprint
    
    # List to store detailed model information
    model_info_list = []
    
    # Build comparison dataframe with evaluation metrics
    comparison_data = []
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        # Get structured model info
        model_type = 'custom' if model_name == 'banana_leaf_cnn' else 'pretrained'
        model_info = get_model_info_from_model(model, model_type=model_type)
        
        # Add evaluation metrics to ModelInfo if available
        if i < len(evaluation_results):
            result = evaluation_results[i]
            model_info.accuracy = result['accuracy']
            model_info.loss = None  # We don't have loss in evaluation results
            
            # Add per-class accuracies if we have them
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
                from cell1_imports_and_constants import IDX_TO_CLASS
                class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
                
                # Calculate per-class accuracies from confusion matrix
                # Diagonal elements are correct predictions for each class
                class_accuracies = {}
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                for class_idx, class_name in enumerate(class_names):
                    if class_idx < len(cm_norm) and cm_norm[class_idx].sum() > 0:
                        class_accuracies[class_name] = cm_norm[class_idx, class_idx]
                    else:
                        class_accuracies[class_name] = 0.0
                
                model_info.class_accuracies = class_accuracies
        
        # Add deployment metrics if available
        if deployment_results and i < len(deployment_results):
            try:
                deploy_df = deployment_results[i]
                # Add inference time if available
                latency_cols = [col for col in deploy_df.columns if 'latency' in col.lower() or 'time' in col.lower()]
                if latency_cols:
                    latency_col = latency_cols[0]
                    model_info.inference_time = deploy_df[latency_col].mean()
            except Exception as e:
                print(f"Warning: Could not add inference time for {model_name}: {e}")
        
        # Add to model info list
        model_info_list.append(model_info)
        
        # Basic model info for main comparison dataframe
        model_data = {
            'Model': model_name,
            'Parameters': model_info.params,
            'Trainable Parameters': model_info.trainable_params,
            'Size (MB)': model_info.model_size_mb
        }
        
        # Add evaluation metrics
        if i < len(evaluation_results):
            model_data.update({
                'Accuracy': evaluation_results[i]['accuracy'],
                'Precision': evaluation_results[i]['precision'],
                'Recall': evaluation_results[i]['recall'],
                'F1 Score': evaluation_results[i]['f1'],
            })
        
        # Add deployment metrics if available
        if deployment_results and i < len(deployment_results):
            try:
                deploy_df = deployment_results[i]
                # Get metrics for batch size 1 (inference speed)
                if 'batch_size' in deploy_df.columns:
                    deploy_row = deploy_df[deploy_df['batch_size'] == 1].iloc[0]
                    inference_time_col = next((col for col in deploy_df.columns if 'inference_time' in col or 'latency' in col), None)
                    memory_usage_col = next((col for col in deploy_df.columns if 'memory' in col), None)
                    
                    if inference_time_col:
                        model_data['Inference Time (ms)'] = deploy_row[inference_time_col]
                    if memory_usage_col:
                        model_data['Memory Usage (MB)'] = deploy_row[memory_usage_col]
                else:
                    # If no batch_size column, use mean values for available metrics
                    inference_time_col = next((col for col in deploy_df.columns if 'inference_time' in col or 'latency' in col), None)
                    memory_usage_col = next((col for col in deploy_df.columns if 'memory' in col), None)
                    
                    if inference_time_col:
                        model_data['Inference Time (ms)'] = deploy_df[inference_time_col].mean()
                    if memory_usage_col:
                        model_data['Memory Usage (MB)'] = deploy_df[memory_usage_col].mean()
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not add deployment metrics for {model_name}: {e}")
        
        # Add energy and carbon footprint estimates if we have training time data
        training_time_seconds = 0
        gpu_used = torch.cuda.is_available()
        
        # Check for training metrics directory
        training_metrics_path = os.path.join(args.output_dir, model_name, 'training', f"{model_name}_resource_metrics.csv")
        if os.path.exists(training_metrics_path):
            try:
                # Load training metrics
                metrics_df = pd.read_csv(training_metrics_path)
                
                # Extract training time
                time_row = metrics_df[metrics_df['Metric'] == 'training_time_seconds']
                if not time_row.empty:
                    training_time_seconds = float(time_row['Value'].iloc[0])
                    
                    # Calculate energy and carbon footprint
                    energy_wh = estimate_energy_consumption(training_time_seconds, gpu_used)
                    carbon_g = calculate_carbon_footprint(energy_wh)
                    
                    # Add to model data
                    model_data['Training Time (s)'] = training_time_seconds
                    model_data['Energy Consumption (Wh)'] = energy_wh
                    model_data['Carbon Footprint (g CO2)'] = carbon_g
            except Exception as e:
                print(f"Warning: Could not estimate energy consumption for {model_name}: {e}")
        
        comparison_data.append(model_data)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comprehensive comparison to CSV
    comprehensive_csv = os.path.join(comparison_dir, "comprehensive_model_comparison.csv")
    comparison_df.to_csv(comprehensive_csv, index=False)
    print(f"Comprehensive model comparison saved to {comprehensive_csv}")
    
    # Generate energy comparison visualizations if we have energy data
    if 'Energy Consumption (Wh)' in comparison_df.columns:
        try:
            print("Generating energy and carbon footprint visualizations...")
            
            # Create energy bar chart
            plt.figure(figsize=(12, 6))
            plt.bar(comparison_df['Model'], comparison_df['Energy Consumption (Wh)'])
            plt.xlabel('Model')
            plt.ylabel('Energy Consumption (Wh)')
            plt.title('Estimated Energy Consumption During Training')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            energy_path = os.path.join(comparison_dir, "energy_consumption_comparison")
            save_figure(plt, energy_path, formats=['png', 'svg'])
            
            # Create carbon footprint bar chart
            plt.figure(figsize=(12, 6))
            plt.bar(comparison_df['Model'], comparison_df['Carbon Footprint (g CO2)'])
            plt.xlabel('Model')
            plt.ylabel('Carbon Footprint (g CO2)')
            plt.title('Estimated Carbon Footprint During Training')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            carbon_path = os.path.join(comparison_dir, "carbon_footprint_comparison")
            save_figure(plt, carbon_path, formats=['png', 'svg'])
            
            # Create scatter plot of energy vs. accuracy
            if 'Accuracy' in comparison_df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(comparison_df['Energy Consumption (Wh)'], comparison_df['Accuracy'] * 100, s=100, alpha=0.7)
                    
                    # Add model names as annotations
                    for i, model_name in enumerate(comparison_df['Model']):
                        plt.annotate(
                            model_name,
                            (comparison_df['Energy Consumption (Wh)'].iloc[i], comparison_df['Accuracy'].iloc[i] * 100),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                    
                    plt.xlabel('Energy Consumption (Wh)')
                    plt.ylabel('Accuracy (%)')
                    plt.title('Trade-off: Accuracy vs. Energy Consumption')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    tradeoff_path = os.path.join(comparison_dir, "accuracy_vs_energy")
                    save_figure(plt, tradeoff_path, formats=['png', 'svg'])
                except Exception as e:
                    print(f"Warning: Could not generate accuracy vs. energy visualization: {e}")
                
            print(f"Energy and carbon footprint visualizations saved to {comparison_dir}")
        except Exception as e:
            print(f"Warning: Could not generate energy visualizations: {e}")
    
    # Save detailed model info as JSON
    model_info_data = [model_info.to_dict() for model_info in model_info_list]
    model_info_df = pd.DataFrame(model_info_data)
    model_info_csv = os.path.join(comparison_dir, "detailed_model_info.csv")
    model_info_df.to_csv(model_info_csv, index=False)
    print(f"Detailed model information saved to {model_info_csv}")
    
    # Generate class-wise accuracy comparison if class accuracies are available
    class_accuracy_data = []
    for model_info in model_info_list:
        if model_info.class_accuracies:
            for class_name, accuracy in model_info.class_accuracies.items():
                class_accuracy_data.append({
                    'Model': model_info.model_name,
                    'Class': class_name,
                    'Accuracy': accuracy
                })
    
    if class_accuracy_data:
        class_acc_df = pd.DataFrame(class_accuracy_data)
        class_acc_csv = os.path.join(comparison_dir, "class_accuracy_comparison.csv")
        class_acc_df.to_csv(class_acc_csv, index=False)
        
        # Generate heatmap of class accuracies
        try:
            plt.figure(figsize=(12, 8))
            class_acc_pivot = class_acc_df.pivot(index='Class', columns='Model', values='Accuracy')
            sns.heatmap(class_acc_pivot, annot=True, cmap='Blues', fmt='.2f')
            plt.title('Class-wise Accuracy Comparison')
            plt.tight_layout()
            
            heatmap_path = os.path.join(comparison_dir, "class_accuracy_heatmap")
            save_figure(plt, heatmap_path, formats=['png', 'svg'])
            print(f"Class-wise accuracy comparison saved to {class_acc_csv}")
        except Exception as e:
            print(f"Warning: Could not generate class accuracy heatmap: {e}")
    
    # Create radar chart for model comparison (if we have at least 2 models)
    if len(model_names) >= 2:
        try:
            # Select numeric columns for radar chart
            numeric_cols = comparison_df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) > 2:  # Need at least 3 axes for a radar chart
                # Normalize values between 0 and 1 for each metric
                norm_df = comparison_df.copy()
                for col in numeric_cols:
                    if 'inference' in col.lower() or 'size' in col.lower() or 'parameters' in col.lower() or 'energy' in col.lower() or 'carbon' in col.lower() or 'time' in col.lower():
                        # For these metrics, lower is better, so invert the normalization
                        max_val = norm_df[col].max()
                        min_val = norm_df[col].min()
                        if max_val > min_val:
                            norm_df[col] = 1 - ((norm_df[col] - min_val) / (max_val - min_val))
                    else:
                        # For these metrics, higher is better
                        max_val = norm_df[col].max()
                        min_val = norm_df[col].min()
                        if max_val > min_val:
                            norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
                
                # Create radar chart
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Select a subset of metrics for clearer visualization
                radar_metrics = ['Accuracy', 'F1 Score', 'Size (MB)', 'Parameters', 'Inference Time (ms)']
                # Add energy metrics if available
                if 'Energy Consumption (Wh)' in comparison_df.columns:
                    radar_metrics.append('Energy Consumption (Wh)')
                
                radar_metrics = [m for m in radar_metrics if m in norm_df.columns]
                
                # Number of variables
                N = len(radar_metrics)
                
                # What will be the angle of each axis in the plot
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Draw the chart for each model
                for i, model_name in enumerate(model_names):
                    values = norm_df.loc[i, radar_metrics].values.tolist()
                    values += values[:1]  # Close the loop
                    
                    # Plot values
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                    ax.fill(angles, values, alpha=0.1)
                
                # Fix axis to go in the right order and start at 12 o'clock
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw axis lines for each angle and label
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_metrics)
                
                # Add legend
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                # Save radar chart
                radar_path = os.path.join(comparison_dir, "model_comparison_radar")
                save_figure(plt, radar_path, formats=['png', 'svg'])
                
                print(f"Radar chart comparison saved to {radar_path}.png")
        except Exception as e:
            print(f"Warning: Could not generate radar chart: {e}")
    
    return comparison_df


def visualize_side_by_side_robustness(robustness_results, output_dir):
    """
    Create side-by-side visualizations comparing robustness of models under different perturbations.
    
    Args:
        robustness_results: List of robustness result dictionaries
        output_dir: Directory to save comparison visualizations
    """
    print("Generating side-by-side robustness comparison visualizations...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names
    model_names = [result['name'] for result in robustness_results]
    
    # Collect perturbation types and accuracy data
    perturbation_types = []
    baseline_accuracies = {}
    accuracy_under_perturbation = {}
    
    for result in robustness_results:
        model_name = result['name']
        results_dict = result['results']
        
        # Get baseline accuracy
        if 'baseline' in results_dict:
            baseline_accuracies[model_name] = results_dict['baseline'].get('accuracy', 0)
        else:
            baseline_accuracies[model_name] = 0
        
        # Get accuracies under perturbation
        for pert_type, pert_result in results_dict.items():
            if pert_type != 'baseline':
                if pert_type not in perturbation_types:
                    perturbation_types.append(pert_type)
                
                if pert_type not in accuracy_under_perturbation:
                    accuracy_under_perturbation[pert_type] = {}
                
                accuracy_under_perturbation[pert_type][model_name] = pert_result.get('accuracy', 0)
    
    # Create a DataFrame for each perturbation type
    comparison_data = []
    
    for pert_type in perturbation_types:
        for model_name in model_names:
            if model_name in accuracy_under_perturbation.get(pert_type, {}):
                pert_acc = accuracy_under_perturbation[pert_type][model_name]
                baseline_acc = baseline_accuracies.get(model_name, 0)
                acc_drop = baseline_acc - pert_acc
                
                comparison_data.append({
                    'Model': model_name,
                    'Perturbation': pert_type.replace('_', ' ').title(),
                    'Baseline Accuracy': baseline_acc,
                    'Perturbed Accuracy': pert_acc,
                    'Accuracy Drop': acc_drop,
                    'Relative Drop (%)': (acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0
                })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "robustness_comparison_detailed.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    # Create visualizations
    # 1. Bar chart showing accuracy drop for each model under each perturbation
    plt.figure(figsize=(14, 8))
    
    # Pivot data for plotting
    pivot_df = comparison_df.pivot(index='Perturbation', columns='Model', values='Accuracy Drop')
    
    # Plot
    ax = pivot_df.plot(kind='bar')
    
    plt.xlabel('Perturbation Type')
    plt.ylabel('Accuracy Drop')
    plt.title('Accuracy Drop Under Different Perturbations')
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    drop_path = os.path.join(output_dir, "accuracy_drop_by_perturbation")
    drop_png, drop_svg = save_figure(plt, drop_path, formats=['png', 'svg'])
    
    # 2. Grouped bar chart showing baseline vs perturbed accuracy
    # For each perturbation type
    for pert_type in perturbation_types:
        pert_data = comparison_df[comparison_df['Perturbation'] == pert_type.replace('_', ' ').title()]
        
        plt.figure(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        baseline_values = [baseline_accuracies.get(model, 0) for model in model_names]
        pert_values = []
        
        for model in model_names:
            model_data = pert_data[pert_data['Model'] == model]
            if len(model_data) > 0:
                pert_values.append(model_data['Perturbed Accuracy'].values[0])
            else:
                pert_values.append(0)
        
        # Plot bars
        ax.bar(x - width/2, baseline_values, width, label='Baseline Accuracy')
        ax.bar(x + width/2, pert_values, width, label=f'Accuracy under {pert_type.replace("_", " ").title()}')
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Model Accuracy Under {pert_type.replace("_", " ").title()} Perturbation')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        pert_path = os.path.join(output_dir, f"accuracy_under_{pert_type}")
        save_figure(plt, pert_path, formats=['png', 'svg'])
    
    # 3. Create heatmap for relative accuracy drop
    plt.figure(figsize=(12, 8))
    pivot_rel_df = comparison_df.pivot(index='Perturbation', columns='Model', values='Relative Drop (%)')
    sns.heatmap(pivot_rel_df, annot=True, cmap='coolwarm_r', fmt='.2f')
    plt.title('Relative Accuracy Drop (%) Under Different Perturbations')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, "relative_drop_heatmap")
    heatmap_png, heatmap_svg = save_figure(plt, heatmap_path, formats=['png', 'svg'])
    
    print(f"Side-by-side robustness comparisons saved to {output_dir}")
    
    return csv_path


def visualize_deployment_comparison(deployment_results, model_names, output_dir):
    """
    Create visualizations comparing deployment metrics across models.
    
    Args:
        deployment_results: List of deployment result dataframes
        model_names: List of model names
        output_dir: Directory to save comparison visualizations
    """
    print("\n📊 Creating deployment metrics comparison visualizations...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a combined dataframe for all models
    combined_data = []
    
    try:
        if not deployment_results:
            print("⚠️ Warning: No deployment results to visualize")
            # Create empty comparison file
            empty_df = pd.DataFrame({'model': model_names})
            empty_csv = os.path.join(output_dir, "deployment_metrics_all_models.csv")
            empty_df.to_csv(empty_csv, index=False)
            return empty_csv
        
        # Check columns in the first dataframe to determine structure
        sample_df = deployment_results[0]
        available_columns = sample_df.columns.tolist()
        print(f"Available columns in deployment metrics: {available_columns}")
        
        # Create a simple combined dataframe with what we have
        for i, (df, model_name) in enumerate(zip(deployment_results, model_names)):
            df_copy = df.copy()
            df_copy['model'] = model_name
            combined_data.append(df_copy)
        
        # Concatenate and save what we have
        combined_df = pd.concat(combined_data, ignore_index=True)
        csv_path = os.path.join(output_dir, "deployment_metrics_all_models.csv")
        combined_df.to_csv(csv_path, index=False)
        
        # Create visualizations for each numeric column
        for col in available_columns:
            if col != 'model' and pd.api.types.is_numeric_dtype(sample_df[col].dtype):
                try:
                    plt.figure(figsize=(10, 6))
                    summary_data = []
                    
                    for df, model_name in zip(deployment_results, model_names):
                        # Use mean value for each metric if column exists
                        summary_data.append({
                            'Model': model_name,
                            col: df[col].mean() if col in df.columns else 0
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    plt.bar(summary_df['Model'], summary_df[col])
                    plt.xlabel('Model')
                    plt.ylabel(col)
                    plt.title(f'{col} Comparison Across Models')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                    # Save figure
                    metric_path = os.path.join(output_dir, f"{col.replace(' ', '_').lower()}_comparison")
                    save_figure(plt, metric_path, formats=['png', 'svg'])
                except Exception as e:
                    print(f"⚠️ Warning: Could not create visualization for {col}: {e}")
        
        # Check for specific columns and create specialized visualizations when available
        
        # Latency visualization
        latency_columns = [col for col in available_columns if 'latency' in col.lower() or 'time' in col.lower()]
        if latency_columns:
            latency_col = latency_columns[0]  # Use the first latency column found
            plt.figure(figsize=(10, 6))
            plt.bar(combined_df['model'], combined_df[latency_col])
            plt.xlabel('Model')
            plt.ylabel('Latency (ms)')
            plt.title('Inference Latency Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            latency_path = os.path.join(output_dir, "inference_latency_comparison")
            save_figure(plt, latency_path, formats=['png', 'svg'])
        
        # Model size visualization
        size_columns = [col for col in available_columns if 'size' in col.lower() and 'mb' in col.lower()]
        if size_columns:
            plt.figure(figsize=(12, 6))
            
            # For each model, plot a group of bars for different size metrics
            x = np.arange(len(model_names))
            width = 0.8 / len(size_columns)
            
            for i, col in enumerate(size_columns):
                values = [df[col].iloc[0] if col in df.columns else 0 for df in deployment_results]
                plt.bar(x + (i - len(size_columns)/2 + 0.5) * width, values, width, label=col)
            
            plt.xlabel('Model')
            plt.ylabel('Size (MB)')
            plt.title('Model Size Comparison by Format')
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            size_path = os.path.join(output_dir, "model_size_comparison")
            save_figure(plt, size_path, formats=['png', 'svg'])
            
        # Throughput visualization
        throughput_columns = [col for col in available_columns if 'throughput' in col.lower() or 'img' in col.lower()]
        if throughput_columns:
            throughput_col = throughput_columns[0]  # Use the first throughput column found
            plt.figure(figsize=(10, 6))
            plt.bar(combined_df['model'], combined_df[throughput_col])
            plt.xlabel('Model')
            plt.ylabel('Throughput (images/second)')
            plt.title('Maximum Throughput Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            throughput_path = os.path.join(output_dir, "throughput_comparison")
            save_figure(plt, throughput_path, formats=['png', 'svg'])
        
        print(f"✅ Deployment metrics comparison saved to {output_dir}")
        return csv_path
        
    except Exception as e:
        print(f"⚠️ Warning: Error generating deployment comparison visualizations: {e}")
        # Create a basic CSV with model names
        empty_df = pd.DataFrame({'model': model_names})
        error_csv = os.path.join(output_dir, "deployment_metrics_error.csv")
        empty_df.to_csv(error_csv, index=False)
        return error_csv


def visualize_efficiency_comparison(models, model_names, evaluation_results, output_dir):
    """
    Create enhanced visualizations comparing efficiency metrics across models.
    
    Args:
        models: List of model objects
        model_names: List of model names
        evaluation_results: List of evaluation result dictionaries
        output_dir: Directory to save comparison visualizations
    """
    print("Generating enhanced efficiency metrics comparison visualizations...")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for each model
    efficiency_data = []
    
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        # Calculate model parameters and size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = calculate_model_size(model)
        
        # Get accuracy from evaluation results
        accuracy = evaluation_results[i]['accuracy'] if i < len(evaluation_results) else 0
        
        # Store data
        efficiency_data.append({
            'Model': model_name,
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Model Size (MB)': model_size_mb,
            'Accuracy': accuracy,
            'Parameter Efficiency': accuracy / (total_params / 1_000_000) if total_params > 0 else 0,  # Accuracy per million parameters
            'Size Efficiency': accuracy / model_size_mb if model_size_mb > 0 else 0  # Accuracy per MB
        })
    
    # Create DataFrame
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "efficiency_metrics_comparison.csv")
    efficiency_df.to_csv(csv_path, index=False)
    
    # Create visualizations
    
    # 1. Parameters comparison
    plt.figure(figsize=(12, 6))
    
    # Sort by total parameters
    sorted_df = efficiency_df.sort_values('Total Parameters')
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to millions for better readability
    total_params_m = sorted_df['Total Parameters'] / 1_000_000
    trainable_params_m = sorted_df['Trainable Parameters'] / 1_000_000
    
    ax.bar(x - width/2, total_params_m, width, label='Total Parameters')
    ax.bar(x + width/2, trainable_params_m, width, label='Trainable Parameters')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Parameters (millions)')
    ax.set_title('Model Parameters Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    params_path = os.path.join(output_dir, "parameters_comparison")
    params_png, params_svg = save_figure(plt, params_path, formats=['png', 'svg'])
    
    # 2. Model size comparison
    plt.figure(figsize=(10, 6))
    
    # Sort by model size
    sorted_by_size = efficiency_df.sort_values('Model Size (MB)')
    
    plt.bar(sorted_by_size['Model'], sorted_by_size['Model Size (MB)'])
    plt.xlabel('Model')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    size_path = os.path.join(output_dir, "model_size_comparison")
    size_png, size_svg = save_figure(plt, size_path, formats=['png', 'svg'])
    
    # 3. Accuracy vs. parameters scatter plot
    plt.figure(figsize=(10, 6))
    
    plt.scatter(
        efficiency_df['Total Parameters'] / 1_000_000,  # Convert to millions
        efficiency_df['Accuracy'] * 100,  # Convert to percentage
        s=100,  # Marker size
        alpha=0.7
    )
    
    # Add labels for each point
    for i, model_name in enumerate(efficiency_df['Model']):
        plt.annotate(
            model_name,
            (efficiency_df['Total Parameters'].iloc[i] / 1_000_000, efficiency_df['Accuracy'].iloc[i] * 100),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Total Parameters (millions)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Model Parameters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    accuracy_params_path = os.path.join(output_dir, "accuracy_vs_parameters")
    accuracy_params_png, accuracy_params_svg = save_figure(plt, accuracy_params_path, formats=['png', 'svg'])
    
    # 4. Parameter efficiency (Accuracy per million parameters)
    plt.figure(figsize=(10, 6))
    
    # Sort by parameter efficiency
    sorted_by_param_eff = efficiency_df.sort_values('Parameter Efficiency', ascending=False)
    
    plt.bar(sorted_by_param_eff['Model'], sorted_by_param_eff['Parameter Efficiency'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy per Million Parameters')
    plt.title('Parameter Efficiency Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    param_eff_path = os.path.join(output_dir, "parameter_efficiency")
    param_eff_png, param_eff_svg = save_figure(plt, param_eff_path, formats=['png', 'svg'])
    
    # 5. Size efficiency (Accuracy per MB)
    plt.figure(figsize=(10, 6))
    
    # Sort by size efficiency
    sorted_by_size_eff = efficiency_df.sort_values('Size Efficiency', ascending=False)
    
    plt.bar(sorted_by_size_eff['Model'], sorted_by_size_eff['Size Efficiency'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy per MB')
    plt.title('Size Efficiency Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    size_eff_path = os.path.join(output_dir, "size_efficiency")
    size_eff_png, size_eff_svg = save_figure(plt, size_eff_path, formats=['png', 'svg'])
    
    # 6. Efficiency radar chart
    if len(model_names) >= 2:
        try:
            # Normalize values between 0 and 1 for radar chart
            radar_df = efficiency_df.copy()
            radar_metrics = ['Accuracy', 'Parameter Efficiency', 'Size Efficiency']
            
            # Normalize each metric
            for metric in radar_metrics:
                max_val = radar_df[metric].max()
                min_val = radar_df[metric].min()
                if max_val > min_val:
                    radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(radar_metrics)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Draw the chart for each model
            for i, model_name in enumerate(model_names):
                values = radar_df.loc[i, radar_metrics].values.tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Efficiency Metrics Radar Chart')
            
            # Save radar chart
            radar_path = os.path.join(output_dir, "efficiency_radar")
            radar_png, radar_svg = save_figure(plt, radar_path, formats=['png', 'svg'])
        except Exception as e:
            print(f"Warning: Could not generate efficiency radar chart: {e}")
    
    print(f"Efficiency metrics comparison visualizations saved to {output_dir}")
    
    return csv_path


def check_disk_space(min_space_mb=100):
    """
    Check if there's enough disk space available.
    
    Args:
        min_space_mb: Minimum required space in MB
        
    Returns:
        True if enough space is available, False otherwise
    """
    try:
        # Get free space in bytes
        free_space = shutil.disk_usage('.').free
        # Convert to MB
        free_space_mb = free_space / (1024 * 1024)
        
        if free_space_mb < min_space_mb:
            print(f"⚠️ Warning: Low disk space! Only {free_space_mb:.2f} MB available. At least {min_space_mb} MB recommended.")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not check disk space: {e}")
        return True  # Assume there's enough space if check fails


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print(f"🌱 BANANA LEAF DISEASE CLASSIFICATION ANALYSIS")
    print("="*80)
    
    if args.all:
        print("Running all analyses")
    else:
        analyses = []
        if args.train: analyses.append("training")
        if args.evaluate: analyses.append("evaluation")
        if args.ablation: analyses.append("ablation studies")
        if args.robustness: analyses.append("robustness testing")
        if args.deployment: analyses.append("deployment metrics")
        print(f"Running analyses: {', '.join(analyses)}")
    
    print(f"Models: {', '.join(args.models)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")
    
    # Set random seed
    try:
        set_seed(args.seed)
    except Exception as e:
        print(f"⚠️ Warning: Could not set random seed: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a dedicated comparisons directory
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Load data
    print("\n📂 LOADING DATA...")
    try:
        train_loader, val_loader, test_loader = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize models
    print("\n🔄 INITIALIZING MODELS...")
    models = []
    model_names = []
    
    # If load_all_models is specified, use the load_pretrained_models function
    # to load all available models for comparison
    if args.load_all_models:
        print("\n📊 Loading all available models for comparison...")
        from cell8_model_zoo import load_pretrained_models
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all available pretrained models
        pretrained_models = load_pretrained_models(device=device)
        
        # Add custom models if specified
        if 'banana_leaf_cnn' in args.models:
            try:
                model = BananaLeafCNN()
                model = model.to(device)
                print(f"✅ Initialized banana_leaf_cnn")
                models.append(model)
                model_names.append('banana_leaf_cnn')
            except Exception as e:
                print(f"❌ Error initializing banana_leaf_cnn: {e}")
        
        # Add pretrained models to our lists
        for name, (model, _) in pretrained_models.items():
            models.append(model)
            model_names.append(name)
            
        print(f"✅ Loaded {len(models)} models for comparison")
    else:
        # Initialize models normally as specified by the user
        for model_name in args.models:
            try:
                if model_name == 'banana_leaf_cnn':
                    model = BananaLeafCNN()
                    print(f"✅ Initialized {model_name}")
                elif model_name == 'mobilenet_v2':
                    model = build_mobilenet_v2(num_classes=NUM_CLASSES)
                    print(f"✅ Initialized {model_name}")
                elif model_name == 'efficientnet_b0':
                    model = build_efficientnet_b0(num_classes=NUM_CLASSES)
                    print(f"✅ Initialized {model_name}")
                elif model_name == 'resnet18':
                    model = build_resnet18(num_classes=NUM_CLASSES)
                    print(f"✅ Initialized {model_name}")
                elif model_name == 'shufflenet_v2':
                    model = build_shufflenet_v2(num_classes=NUM_CLASSES)
                    print(f"✅ Initialized {model_name}")
                else:
                    # Other models from the model zoo
                    model_adapter, _ = create_model_adapter(model_name, pretrained=True, freeze_backbone=False)
                    if model_adapter is None:
                        print(f"⚠️ Warning: Failed to create model {model_name}. Skipping.")
                        continue
                    model = model_adapter
                    print(f"✅ Initialized {model_name}")
                
                models.append(model)
                model_names.append(model_name)
            except Exception as e:
                print(f"❌ Error initializing model {model_name}: {e}")
    
    if not models:
        print("❌ No models were initialized. Exiting.")
        return
    
    # Train models
    training_metrics = []
    if args.train or args.all:
        print("\n" + "="*80)
        print("🏋️ TRAINING MODELS")
        print("="*80)
        
        for model, model_name in zip(models, model_names):
            try:
                print(f"\n📊 Training {model_name}...")
                metrics = train_model(model, model_name, train_loader, val_loader, args)
                training_metrics.append(metrics)
                print(f"✅ Training completed for {model_name}")
            except Exception as e:
                print(f"❌ Error training model {model_name}: {e}")
                training_metrics.append(None)
        
        # Compare training resources
        if len(models) > 1 and any(training_metrics):
            try:
                print("\n📊 Generating training resource comparison visualizations...")
                filtered_metrics = [m for m in training_metrics if m is not None]
                csv_path, plot_path = compare_training_resources(
                    filtered_metrics
                )
                print(f"✅ Training resource comparison saved to {csv_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not compare training resources: {e}")
    else:
        # Load pre-trained models
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            checkpoint_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
            try:
                if os.path.exists(checkpoint_path):
                    print(f"📂 Loading pre-trained model {model_name} from {checkpoint_path}")
                    model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
                    models[i] = model
                    print(f"✅ Model {model_name} loaded successfully")
                else:
                    print(f"⚠️ Warning: No pre-trained model found for {model_name}. Using untrained model.")
            except Exception as e:
                print(f"⚠️ Warning: Could not load model {model_name}: {e}")
    
    # Evaluate models
    evaluation_results = []
    if args.evaluate or args.all:
        print("\n" + "="*80)
        print("📊 EVALUATING MODELS")
        print("="*80)
        
        try:
            evaluation_results = evaluate_models(models, model_names, test_loader, args)
            print(f"✅ Evaluation completed for {len(evaluation_results)} models")
        except Exception as e:
            print(f"❌ Error during model evaluation: {e}")
    else:
        # Create dummy evaluation results
        for model_name in model_names:
            evaluation_results.append({
                'name': model_name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'y_true': [],
                'y_pred': [],
                'confusion_matrix': []
            })
    
    # Run ablation studies
    ablation_results = []
    if args.ablation or args.all:
        print("\n" + "="*80)
        print("🧪 RUNNING ABLATION STUDIES")
        print("="*80)
        
        # Get trained models (first try to load checkpoints)
        trained_models = []
        trained_model_names = []
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            try:
                checkpoint_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
                if os.path.exists(checkpoint_path):
                    print(f"📂 Using trained model {model_name} from {checkpoint_path} for ablation")
                    model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
                    trained_models.append(model)
                    trained_model_names.append(model_name)
                elif args.train or args.all:
                    # If we just trained this model, use it
                    print(f"📊 Using freshly trained model {model_name} for ablation")
                    trained_models.append(model)
                    trained_model_names.append(model_name)
                else:
                    print(f"⚠️ Skipping ablation for {model_name} (no trained model found)")
            except Exception as e:
                print(f"⚠️ Warning: Error loading model {model_name} for ablation: {e}")
        
        # Only run ablation on trained models
        if not trained_models:
            print("❌ No trained models available for ablation studies")
        else:
            ablation_epochs = min(10, args.epochs)  # Use fewer epochs for ablation
            for model, model_name in zip(trained_models, trained_model_names):
                try:
                    print(f"Running ablation studies for {model_name}...")
                    # Create ablation study with fewer epochs
                    ablation_study = AblationStudy(
                        base_model=model,
                        model_name=model_name,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=ablation_epochs,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        output_dir=os.path.join(args.output_dir, 'ablation')
                    )
                    
                    # Create model variants
                    variants = [
                        create_model_variant(
                            model, 
                            lambda m: change_dropout_rate(m, 0.3),
                            f"{model_name}_dropout_0.3",
                            "Dropout rate changed to 0.3"
                        ),
                        create_model_variant(
                            model, 
                            lambda m: change_dropout_rate(m, 0.7),
                            f"{model_name}_dropout_0.7",
                            "Dropout rate changed to 0.7"
                        ),
                        create_model_variant(
                            model, 
                            lambda m: change_activation(m, 'leaky_relu'),
                            f"{model_name}_leaky_relu",
                            "ReLU activation changed to LeakyReLU"
                        ),
                        # NEW: Try different normalization types
                        create_model_variant(
                            model,
                            lambda m: change_normalization(m, 'instance'),
                            f"{model_name}_instance_norm",
                            "Batch normalization changed to instance normalization"
                        ),
                        create_model_variant(
                            model,
                            lambda m: change_normalization(m, 'layer'),
                            f"{model_name}_group_norm",
                            "Batch normalization changed to group normalization (one group per channel)"
                        ),
                        # NEW: Try removing a layer (this requires model-specific knowledge)
                        # For BananaLeafCNN model
                        create_model_variant(
                            model,
                            lambda m: remove_layer(m, 'dropout') if model_name == 'banana_leaf_cnn' else m,
                            f"{model_name}_no_dropout",
                            "Dropout layers removed" if model_name == 'banana_leaf_cnn' 
                            else "Layer removal not applicable"
                        )
                    ]
                    
                    # Run ablation studies
                    results_df = ablation_study.run_ablation_studies(variants)
                    ablation_results.append({'name': model_name, 'results': results_df})
                    print(f"✅ Ablation studies completed for {model_name}")
                except Exception as e:
                    print(f"❌ Error running ablation studies for {model_name}: {e}")
            
            # Generate ablation comparison across models if we have multiple models
            if len(ablation_results) > 1:
                try:
                    print("\n📊 Generating ablation comparison across models...")
                    # Create a comparison dataframe
                    ablation_comparison = pd.DataFrame()
                    
                    for result in ablation_results:
                        model_name = result['name']
                        model_df = result['results'].copy()
                        
                        # Ensure variant_name column exists (it might be named 'variant' in some versions)
                        if 'variant' in model_df.columns and 'variant_name' not in model_df.columns:
                            model_df['variant_name'] = model_df['variant']
                        
                        model_df['model'] = model_name
                        if ablation_comparison.empty:
                            ablation_comparison = model_df
                        else:
                            ablation_comparison = pd.concat([ablation_comparison, model_df], ignore_index=True)
                    
                    # Save comparison to CSV
                    ablation_comparison_csv = os.path.join(comparisons_dir, "ablation_comparison.csv")
                    ablation_comparison.to_csv(ablation_comparison_csv, index=False)
                    
                    # Generate comparison visualization
                    plt.figure(figsize=(12, 8))
                    
                    for model_name in trained_model_names:
                        model_data = ablation_comparison[ablation_comparison['model'] == model_name]
                        plt.plot(model_data['variant_name'], model_data['val_accuracy'], 
                                 marker='o', linestyle='-', label=model_name)
                    
                    plt.xlabel('Model Variant')
                    plt.ylabel('Validation Accuracy')
                    plt.title('Ablation Study Comparison Across Models')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    
                    # Save figure
                    ablation_comparison_plot = os.path.join(comparisons_dir, "ablation_comparison")
                    save_figure(plt, ablation_comparison_plot, formats=['png', 'svg'])
                    
                    # NEW: Enhanced visualization - Group variants by type
                    variant_categories = {
                        'dropout': [v for v in ablation_comparison['variant_name'] if 'dropout' in v],
                        'activation': [v for v in ablation_comparison['variant_name'] if 'relu' in v or 'sigmoid' in v or 'tanh' in v],
                        'normalization': [v for v in ablation_comparison['variant_name'] if 'norm' in v],
                        'layer_removal': [v for v in ablation_comparison['variant_name'] if 'no_' in v]
                    }
                    
                    # Ablation impact analysis - effect on accuracy by category
                    plt.figure(figsize=(14, 10))
                    
                    # Create subplots for each category
                    fig, axs = plt.subplots(len(variant_categories), 1, figsize=(14, 4*len(variant_categories)))
                    
                    # If only one category, make axs iterable
                    if len(variant_categories) == 1:
                        axs = [axs]
                        
                    for i, (category, variants) in enumerate(variant_categories.items()):
                        # Skip empty categories
                        if not variants:
                            continue
                            
                        # Filter data for this category
                        category_data = ablation_comparison[ablation_comparison['variant_name'].isin(variants)]
                        if category_data.empty:
                            continue
                            
                        # Plot for each model
                        for model_name in trained_model_names:
                            model_data = category_data[category_data['model'] == model_name]
                            if not model_data.empty:
                                axs[i].plot(model_data['variant_name'], model_data['val_accuracy'], 
                                         marker='o', linestyle='-', label=model_name)
                                
                        # Find baseline accuracy for each model
                        for model_name in trained_model_names:
                            base_data = ablation_comparison[
                                (ablation_comparison['model'] == model_name) & 
                                (ablation_comparison['variant_name'] == model_name)
                            ]
                            if not base_data.empty:
                                base_acc = base_data['val_accuracy'].values[0]
                                axs[i].axhline(y=base_acc, color='gray', linestyle='--', 
                                            label=f'{model_name} Baseline' if model_name == trained_model_names[0] else None)
                        
                        axs[i].set_title(f'Effect of {category.replace("_", " ").title()} Modifications')
                        axs[i].set_xlabel('Variant')
                        axs[i].set_ylabel('Validation Accuracy')
                        axs[i].tick_params(axis='x', rotation=45)
                        axs[i].legend()
                        axs[i].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    ablation_category_plot = os.path.join(comparisons_dir, "ablation_by_category")
                    save_figure(plt, ablation_category_plot, formats=['png', 'svg'])
                    
                    # NEW: Radar chart comparing different aspects of modifications
                    if len(trained_model_names) > 0:
                        try:
                            plt.figure(figsize=(10, 10))
                            
                            # Convert to relative performance (% change from baseline)
                            relative_performance = []
                            
                            for model_name in trained_model_names:
                                # Get baseline accuracy
                                base_data = ablation_comparison[
                                    (ablation_comparison['model'] == model_name) & 
                                    (ablation_comparison['variant_name'] == model_name)
                                ]
                                
                                if not base_data.empty:
                                    base_acc = base_data['val_accuracy'].values[0]
                                    
                                    # Calculate relative performance for each variant
                                    for _, row in ablation_comparison[ablation_comparison['model'] == model_name].iterrows():
                                        if row['variant_name'] != model_name:  # Skip baseline
                                            rel_change = (row['val_accuracy'] - base_acc) / base_acc * 100
                                            
                                            # Determine the category
                                            category = None
                                            for cat, variants in variant_categories.items():
                                                if row['variant_name'] in variants:
                                                    category = cat
                                                    break
                                            
                                            if category:
                                                relative_performance.append({
                                                    'model': model_name,
                                                    'variant': row['variant_name'],
                                                    'category': category,
                                                    'relative_change': rel_change
                                                })
                            
                            if relative_performance:
                                rel_df = pd.DataFrame(relative_performance)
                                
                                # Save to CSV
                                rel_csv = os.path.join(comparisons_dir, "ablation_relative_performance.csv")
                                rel_df.to_csv(rel_csv, index=False)
                                
                                # Generate heatmap of relative changes
                                plt.figure(figsize=(12, 8))
                                pivot_df = rel_df.pivot(index='variant', columns='model', values='relative_change')
                                sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
                                plt.title('Relative Change in Accuracy (%)')
                                plt.tight_layout()
                                
                                # Save heatmap
                                heatmap_path = os.path.join(comparisons_dir, "ablation_relative_heatmap")
                                save_figure(plt, heatmap_path, formats=['png', 'svg'])
                                
                                print(f"✅ Enhanced ablation visualizations saved to {comparisons_dir}")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not generate enhanced ablation visualizations: {e}")
                    
                    print(f"✅ Ablation comparison saved to {ablation_comparison_csv}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate ablation comparison: {e}")
    
    # Run robustness tests
    robustness_results = []
    if args.robustness or args.all:
        print("\n" + "="*80)
        print("🛡️ RUNNING ROBUSTNESS TESTS")
        print("="*80)
        
        for model, model_name in zip(models, model_names):
            try:
                print(f"\n📊 Running robustness tests for {model_name}...")
                test_results = run_robustness_tests(model, model_name, test_loader, args)
                robustness_results.append({'name': model_name, 'model': model, 'results': test_results})
                print(f"✅ Robustness tests completed for {model_name}")
            except Exception as e:
                print(f"❌ Error running robustness tests for {model_name}: {e}")
        
        # Compare model robustness if we have multiple models
        if len(robustness_results) > 1:
            print("\n📊 Comparing model robustness...")
            
            # Create a directory for comparison results
            robustness_comparison_dir = os.path.join(comparisons_dir, 'robustness')
            os.makedirs(robustness_comparison_dir, exist_ok=True)
            
            # Generate comparisons for each perturbation type
            perturbation_types = ['gaussian_noise', 'blur', 'brightness', 'contrast', 'rotation', 'occlusion', 'jpeg_compression']
            for perturbation_type in perturbation_types:
                try:
                    print(f"📊 Generating comparison for {perturbation_type}...")
                    comparison_df, comparison_csv, comparison_plot = compare_model_robustness(
                        [{'name': r['name'], 'model': r['model']} for r in robustness_results],
                        test_loader,
                        perturbation_type=perturbation_type,
                        output_dir=robustness_comparison_dir
                    )
                    print(f"✅ Robustness comparison for {perturbation_type} saved to {comparison_csv}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate robustness comparison for {perturbation_type}: {e}")
            
            # Create overall robustness summary
            try:
                print("\n📊 Generating overall robustness summary...")
                
                # Check if robustness_results is empty
                if not robustness_results:
                    print("⚠️ Warning: No robustness results available for visualization")
                    return
                
                # Generate side-by-side comparisons of all models under all perturbations
                visualize_side_by_side_robustness(robustness_results, robustness_comparison_dir)
                
                # Prepare data for summary plot
                summary_data = []
                
                for result in robustness_results:
                    model_name = result['name']
                    model_results = result['results']
                    
                    # Check if model_results has the expected structure
                    if not isinstance(model_results, dict) or 'baseline' not in model_results:
                        print(f"⚠️ Warning: Invalid robustness results format for {model_name}")
                        continue
                    
                    # Calculate average accuracy drop across all perturbations
                    baseline_acc = model_results.get('baseline', {}).get('accuracy', 0)
                    
                    for pert_type, pert_result in model_results.items():
                        if pert_type != 'baseline' and isinstance(pert_result, dict):
                            pert_acc = pert_result.get('accuracy', 0)
                            
                            # Skip if metrics are not numeric
                            if not isinstance(baseline_acc, (int, float)) or not isinstance(pert_acc, (int, float)):
                                print(f"⚠️ Warning: Non-numeric accuracy values for {model_name}, {pert_type}")
                                continue
                            
                            acc_drop = baseline_acc - pert_acc
                            
                            summary_data.append({
                                'Model': model_name,
                                'Perturbation': pert_type.replace('_', ' ').title(),
                                'Accuracy Drop': acc_drop
                            })
                
                # Create summary dataframe if we have data
                if not summary_data:
                    print("⚠️ Warning: No valid robustness data available for visualization")
                    return
                    
                summary_df = pd.DataFrame(summary_data)
                
                # Save to CSV
                summary_csv = os.path.join(robustness_comparison_dir, "robustness_summary.csv")
                summary_df.to_csv(summary_csv, index=False)
                
                # Verify we have enough unique models and perturbations for a meaningful heatmap
                unique_models = summary_df['Model'].unique()
                unique_perturbations = summary_df['Perturbation'].unique()
                
                if len(unique_models) < 1 or len(unique_perturbations) < 1:
                    print("⚠️ Warning: Not enough unique models or perturbations for heatmap")
                    return
                
                # Create pivot table with explicit handling for missing values
                try:
                    # Create heatmap for accuracy drop
                    plt.figure(figsize=(12, 8))
                    
                    # Ensure pivot works even with missing data
                    pivot_df = summary_df.pivot_table(
                        index='Perturbation', 
                        columns='Model', 
                        values='Accuracy Drop',
                        aggfunc='mean',  # Use mean in case of duplicates
                        fill_value=0     # Fill missing with zeros
                    )
                    
                    # Check if pivot is empty
                    if pivot_df.empty or pivot_df.isnull().all().all():
                        print("⚠️ Warning: Empty pivot table for heatmap")
                        return
                    
                    sns.heatmap(pivot_df, annot=True, cmap='coolwarm_r', fmt='.3f')
                    plt.title('Accuracy Drop Under Different Perturbations')
                    plt.tight_layout()
                    
                    # Save heatmap
                    heatmap_path = os.path.join(robustness_comparison_dir, "robustness_heatmap")
                    save_figure(plt, heatmap_path, formats=['png', 'svg'])
                    
                    print(f"✅ Robustness summary saved to {summary_csv}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to create robustness heatmap: {e}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not generate robustness summary: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
    
    # Run deployment metrics
    deployment_results = []
    if args.deployment or args.all:
        print("\n" + "="*80)
        print("🚀 RUNNING DEPLOYMENT METRICS")
        print("="*80)
        
        for model, model_name in zip(models, model_names):
            try:
                print(f"\n📊 Running deployment metrics analysis for {model_name}...")
                metrics = run_deployment_metrics(model, model_name, args)
                deployment_results.append(metrics)
                print(f"✅ Deployment metrics analysis completed for {model_name}")
            except Exception as e:
                print(f"❌ Error analyzing deployment metrics for {model_name}: {e}")
        
        # Compare deployment metrics
        if len(deployment_results) > 1:
            try:
                print("\n📊 Comparing deployment metrics...")
                
                # Create deployment comparisons directory
                deployment_comparison_dir = os.path.join(comparisons_dir, 'deployment')
                os.makedirs(deployment_comparison_dir, exist_ok=True)
                
                # Create standard deployment metrics comparison using existing function
                model_list = [{'name': name, 'model': model} for name, model in zip(model_names, models)]
                df, csv_path, plot_path = compare_deployment_metrics(
                    model_list,
                    output_dir=deployment_comparison_dir
                )
                
                # Create enhanced visualizations for deployment comparison
                visualize_deployment_comparison(deployment_results, model_names, deployment_comparison_dir)
                
                print(f"✅ Deployment metrics comparison saved to {deployment_comparison_dir}")
            except Exception as e:
                print(f"⚠️ Warning: Could not compare deployment metrics: {e}")
    
    # Run efficiency metrics
    efficiency_results = None
    if args.all or len(models) > 1:  # Always run efficiency metrics when comparing multiple models
        print("\n" + "="*80)
        print("⚡ RUNNING EFFICIENCY METRICS")
        print("="*80)
        
        try:
            # Create efficiency comparisons directory
            efficiency_comparison_dir = os.path.join(comparisons_dir, 'efficiency')
            os.makedirs(efficiency_comparison_dir, exist_ok=True)
            
            # Run standard efficiency metrics
            print("\n📊 Calculating efficiency metrics...")
            efficiency_results = run_efficiency_metrics(models, model_names, evaluation_results, args)
            print("✅ Efficiency metrics calculated")
            
            # Generate enhanced efficiency visualizations for direct model comparison
            if len(models) > 1:
                print("\n📊 Generating enhanced efficiency visualizations...")
                visualize_efficiency_comparison(models, model_names, evaluation_results, efficiency_comparison_dir)
                print(f"✅ Enhanced efficiency visualizations saved to {efficiency_comparison_dir}")
        except Exception as e:
            print(f"❌ Error analyzing efficiency metrics: {e}")
    
    # Generate comprehensive comparison summary
    if len(models) > 1:
        try:
            print("\n📊 Generating comprehensive model comparison...")
            comprehensive_df = save_comprehensive_comparison(
                models, 
                model_names, 
                evaluation_results, 
                deployment_results,
                efficiency_results,
                robustness_results,
                args
            )
            print("✅ Comprehensive comparison generated")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate comprehensive comparison: {e}")
    
    # Generate per-class performance comparison
    if (args.evaluate or args.all) and len(models) > 1:
        try:
            print("\n📊 Generating per-class performance comparison...")
            from cell1_imports_and_constants import IDX_TO_CLASS
            
            # Prepare data for per-class comparison
            class_names = list(IDX_TO_CLASS.values())
            model_names_list = model_names.copy()
            
            # Calculate per-class metrics for each model
            per_class_data = {}
            
            # Check if we have any evaluation results with predictions
            valid_results = [r for r in evaluation_results if r.get('y_true') and r.get('y_pred')]
            if not valid_results:
                print("⚠️ Warning: No valid evaluation results with predictions found for per-class comparison")
                return
            
            for result in valid_results:
                model_name = result['name']
                y_true = result['y_true']
                y_pred = result['y_pred']
                
                # Skip if y_true or y_pred are empty
                if not y_true or not y_pred:
                    print(f"⚠️ Warning: Empty prediction data for {model_name}, skipping in per-class comparison")
                    continue
                
                # Calculate per-class accuracy
                per_class_acc = []
                for class_idx in range(len(class_names)):
                    # Find indices where true label is this class
                    indices = [i for i, label in enumerate(y_true) if label == class_idx]
                    
                    if indices:
                        # Calculate accuracy for this class
                        correct = sum(1 for i in indices if y_pred[i] == y_true[i])
                        class_acc = correct / len(indices)
                    else:
                        class_acc = 0
                    
                    per_class_acc.append(class_acc)
                
                per_class_data[model_name] = per_class_acc
            
            # Check if we have any data to plot
            if not per_class_data:
                print("⚠️ Warning: No valid per-class data available for visualization")
                return
            
            # Convert to DataFrame
            per_class_df = pd.DataFrame(per_class_data, index=class_names)
            
            # Save to CSV
            per_class_csv = os.path.join(comparisons_dir, "per_class_accuracy_comparison.csv")
            per_class_df.to_csv(per_class_csv)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(per_class_df, annot=True, cmap='Blues', fmt='.2f')
            plt.title('Per-Class Accuracy Comparison')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(comparisons_dir, "per_class_accuracy_heatmap")
            save_figure(plt, heatmap_path, formats=['png', 'svg'])
            
            # Create bar chart for each class
            for class_idx, class_name in enumerate(class_names):
                plt.figure(figsize=(10, 6))
                # Check if all required model names are in per_class_data before creating chart
                available_models = [m for m in model_names_list if m in per_class_data]
                if not available_models:
                    print(f"⚠️ Warning: No models available for class {class_name}, skipping chart")
                    plt.close()
                    continue
                
                class_accuracies = [per_class_data[model_name][class_idx] for model_name in available_models]
                plt.bar(available_models, class_accuracies)
                plt.xlabel('Model')
                plt.ylabel('Accuracy')
                plt.title(f'Accuracy for Class: {class_name}')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                class_path = os.path.join(comparisons_dir, f"class_{class_name.replace(' ', '_')}_comparison")
                save_figure(plt, class_path, formats=['png', 'svg'])
            
            print(f"✅ Per-class comparison saved to {per_class_csv}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate per-class comparison: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"Results saved to {args.output_dir}")
    print(f"Comparison results saved to {comparisons_dir}")
    print("="*80)


if __name__ == "__main__":
    main() 