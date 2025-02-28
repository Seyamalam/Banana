import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import modules
from cell1_imports_and_constants import set_seed, NUM_CLASSES
from cell2_dataset import load_data
from cell3_model import BananaLeafCNN
from cell4_training import train, validate
from cell6_utils import save_checkpoint, load_checkpoint, evaluate_model
from cell11_training_resources import measure_training_resources, compare_training_resources
from cell12_statistical_testing import statistical_significance_test, mcnemar_test
from cell13_efficiency_metrics import calculate_advanced_efficiency_metrics, calculate_pareto_frontier, calculate_model_size
from cell14_ablation_studies import AblationStudy, create_model_variant, change_dropout_rate, change_activation
from cell15_flops_analysis import calculate_flops, compare_model_flops, analyze_layer_distribution
from cell16_robustness_testing import RobustnessTest, compare_model_robustness
from cell17_cross_dataset import evaluate_cross_dataset, compare_cross_dataset_performance
from cell18_deployment_metrics import benchmark_deployment_metrics, compare_deployment_metrics

# Import model zoo for comparison
from cell8_model_zoo import (
    build_mobilenet_v2, 
    build_efficientnet_b0, 
    build_resnet18, 
    build_shufflenet_v2,
    get_available_classification_models,
    create_model_adapter
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
        plot_class_distribution(all_labels, save_path=dist_path, formats=['png', 'svg'])
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
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                outputs = model(images.to(device))
                _, preds = torch.max(outputs, 1)
            
            # Visualize predictions
            vis_path = os.path.join(eval_dir, f"{model_name}_predictions")
            visualize_predictions(images[:16], labels[:16], preds[:16].cpu(), vis_path, formats=['png', 'svg'])
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
    
    # Statistical significance testing
    if len(results) > 1:
        print("\nPerforming statistical significance testing...")
        p_value_df, csv_path, plot_path = statistical_significance_test(results, output_dir=args.output_dir)
        print(f"Statistical significance results saved to {csv_path}")
        
        # McNemar test
        mcnemar_df, mcnemar_csv = mcnemar_test(results, output_dir=args.output_dir)
        print(f"McNemar test results saved to {mcnemar_csv}")
    
    return results


def run_robustness_tests(model, model_name, test_loader, args):
    """Run robustness tests on model."""
    print(f"Running robustness tests for {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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


def run_deployment_metrics(model, model_name, args):
    """Run deployment metrics analysis."""
    print(f"Running deployment metrics analysis for {model_name}...")
    
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


def run_efficiency_metrics(models, model_names, results, args):
    """Run efficiency metrics analysis."""
    print("Running efficiency metrics analysis...")
    
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
            'accuracy': results[i]['accuracy'],
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
    pareto_df, pareto_csv, pareto_plot = calculate_pareto_frontier(
        model_results,
        output_dir=os.path.join(args.output_dir, 'efficiency')
    )
    
    print(f"Pareto frontier analysis saved to {pareto_csv}")
    
    # Compare FLOPs
    flops_df, flops_csv, flops_plot = compare_model_flops(
        model_results,
        output_dir=os.path.join(args.output_dir, 'efficiency')
    )
    
    print(f"FLOPs comparison saved to {flops_csv}")
    
    return metrics_df


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print(f"Starting Banana Leaf Disease Classification Analysis")
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
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Initialize models
    models = []
    model_names = []
    
    for model_name in args.models:
        if model_name == 'banana_leaf_cnn':
            model = BananaLeafCNN()
        elif model_name == 'mobilenet_v2':
            model = build_mobilenet_v2(num_classes=NUM_CLASSES)
        elif model_name == 'efficientnet_b0':
            model = build_efficientnet_b0(num_classes=NUM_CLASSES)
        elif model_name == 'resnet18':
            model = build_resnet18(num_classes=NUM_CLASSES)
        elif model_name == 'shufflenet_v2':
            model = build_shufflenet_v2(num_classes=NUM_CLASSES)
        else:
            # Other models from the model zoo
            model_adapter, _ = create_model_adapter(model_name, pretrained=True, freeze_backbone=False)
            if model_adapter is None:
                print(f"Warning: Failed to create model {model_name}. Skipping.")
                continue
            model = model_adapter
        
        models.append(model)
        model_names.append(model_name)
    
    # Train models
    if args.train or args.all:
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        training_metrics = []
        
        for model, model_name in zip(models, model_names):
            metrics = train_model(model, model_name, train_loader, val_loader, args)
            training_metrics.append(metrics)
        
        # Compare training resources
        if len(training_metrics) > 1:
            print("Generating training resource comparison visualizations...")
            csv_path, plot_path = compare_training_resources(
                training_metrics,
                output_dir=os.path.join(args.output_dir, 'comparisons')
            )
            print(f"Training resource comparison saved to {csv_path}")
    else:
        # Load pre-trained models
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            checkpoint_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
            if os.path.exists(checkpoint_path):
                print(f"Loading pre-trained model {model_name} from {checkpoint_path}")
                model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
                models[i] = model
            else:
                print(f"Warning: No pre-trained model found for {model_name}. Using untrained model.")
    
    # Evaluate models
    if args.evaluate or args.all:
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)
        evaluation_results = evaluate_models(models, model_names, test_loader, args)
    else:
        # Create dummy evaluation results
        evaluation_results = []
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
    if args.ablation or args.all:
        print("\n" + "="*80)
        print("RUNNING ABLATION STUDIES")
        print("="*80)
        
        # Get trained models (first try to load checkpoints)
        trained_models = []
        trained_model_names = []
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            checkpoint_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
            if os.path.exists(checkpoint_path):
                print(f"Using trained model {model_name} from {checkpoint_path} for ablation")
                model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
                trained_models.append(model)
                trained_model_names.append(model_name)
            elif args.train or args.all:
                # If we just trained this model, use it
                print(f"Using freshly trained model {model_name} for ablation")
                trained_models.append(model)
                trained_model_names.append(model_name)
            else:
                print(f"Skipping ablation for {model_name} (no trained model found)")
        
        # Only run ablation on trained models
        ablation_epochs = min(10, args.epochs)  # Use fewer epochs for ablation
        for model, model_name in zip(trained_models, trained_model_names):
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
                )
            ]
            
            # Run ablation studies
            results_df = ablation_study.run_ablation_studies(variants)
            print(f"Ablation study results saved to {os.path.join(args.output_dir, 'ablation')}")
    
    # Run robustness tests
    if args.robustness or args.all:
        print("\n" + "="*80)
        print("RUNNING ROBUSTNESS TESTS")
        print("="*80)
        robustness_results = []
        for model, model_name in zip(models, model_names):
            test_results = run_robustness_tests(model, model_name, test_loader, args)
            robustness_results.append({'name': model_name, 'model': model, 'results': test_results})
        
        # Compare model robustness if we have multiple models
        if len(robustness_results) > 1:
            print("\nComparing model robustness...")
            for perturbation_type in ['gaussian_noise', 'blur', 'brightness', 'contrast', 'rotation', 'occlusion', 'jpeg_compression']:
                try:
                    comparison_df, comparison_csv, comparison_plot = compare_model_robustness(
                        [{'name': r['name'], 'model': r['model']} for r in robustness_results],
                        test_loader,
                        perturbation_type=perturbation_type,
                        output_dir=os.path.join(args.output_dir, 'robustness', 'comparisons')
                    )
                    print(f"Robustness comparison for {perturbation_type} saved to {comparison_csv}")
                except Exception as e:
                    print(f"Warning: Could not generate robustness comparison for {perturbation_type}: {e}")
    
    # Run deployment metrics
    if args.deployment or args.all:
        print("\n" + "="*80)
        print("RUNNING DEPLOYMENT METRICS")
        print("="*80)
        deployment_results = []
        for model, model_name in zip(models, model_names):
            metrics = run_deployment_metrics(model, model_name, args)
            deployment_results.append(metrics)
        
        # Compare deployment metrics
        if len(deployment_results) > 1:
            print("\nComparing deployment metrics...")
            model_list = [{'name': name, 'model': model} for name, model in zip(model_names, models)]
            df, csv_path, plot_path = compare_deployment_metrics(
                model_list,
                output_dir=os.path.join(args.output_dir, 'deployment', 'comparisons')
            )
            print(f"Deployment metrics comparison saved to {csv_path}")
    
    # Run efficiency metrics
    if args.all:
        print("\n" + "="*80)
        print("RUNNING EFFICIENCY METRICS")
        print("="*80)
        efficiency_results = run_efficiency_metrics(models, model_names, evaluation_results, args)
    
    # Generate final comparison summary
    if len(models) > 1:
        print("\n" + "="*80)
        print("GENERATING OVERALL MODEL COMPARISON")
        print("="*80)
        
        # Create comparison directory
        comparison_dir = os.path.join(args.output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Build comparison dataframe with key metrics
        comparison_data = []
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            model_data = {
                'Model': model_name,
                'Accuracy': evaluation_results[i]['accuracy'] if i < len(evaluation_results) else 0,
                'F1 Score': evaluation_results[i]['f1'] if i < len(evaluation_results) else 0,
                'Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'Size (MB)': calculate_model_size(model)
            }
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_csv = os.path.join(comparison_dir, "model_comparison_summary.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        
        # Create bar charts for key metrics
        metrics_to_plot = ['Accuracy', 'F1 Score', 'Parameters', 'Size (MB)']
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            plt.bar(comparison_df['Model'], comparison_df[metric])
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.title(f'Comparison of {metric} Across Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            metric_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(comparison_dir, f"comparison_{metric_name}")
            save_figure(plt, save_path, formats=['png', 'svg'])
        
        print(f"Overall model comparison saved to {comparison_csv}")
        
        # Add per-class performance comparison
        if args.evaluate or args.all:
            try:
                print("\nGenerating per-class performance comparison...")
                from cell1_imports_and_constants import IDX_TO_CLASS
                
                # Prepare data for per-class comparison
                class_names = list(IDX_TO_CLASS.values())
                model_names_list = model_names.copy()
                
                # Calculate per-class metrics for each model
                per_class_data = {}
                
                for result in evaluation_results:
                    model_name = result['name']
                    y_true = result['y_true']
                    y_pred = result['y_pred']
                    
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
                
                # Convert to DataFrame
                per_class_df = pd.DataFrame(per_class_data, index=class_names)
                
                # Save to CSV
                per_class_csv = os.path.join(comparison_dir, "per_class_accuracy_comparison.csv")
                per_class_df.to_csv(per_class_csv)
                
                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(per_class_df, annot=True, cmap='Blues', fmt='.2f')
                plt.title('Per-Class Accuracy Comparison')
                plt.tight_layout()
                
                # Save heatmap
                heatmap_path = os.path.join(comparison_dir, "per_class_accuracy_heatmap")
                save_figure(plt, heatmap_path, formats=['png', 'svg'])
                
                # Create bar chart for each class
                for class_idx, class_name in enumerate(class_names):
                    plt.figure(figsize=(10, 6))
                    class_accuracies = [per_class_data[model_name][class_idx] for model_name in model_names_list]
                    plt.bar(model_names_list, class_accuracies)
                    plt.xlabel('Model')
                    plt.ylabel('Accuracy')
                    plt.title(f'Accuracy for Class: {class_name}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save figure
                    class_path = os.path.join(comparison_dir, f"class_{class_name.replace(' ', '_')}_comparison")
                    save_figure(plt, class_path, formats=['png', 'svg'])
                
                print(f"Per-class comparison saved to {per_class_csv}")
            except Exception as e:
                print(f"Warning: Could not generate per-class comparison: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main() 