import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
import torch
from sklearn.metrics import accuracy_score
from cell5_visualization import save_figure


def bootstrap_sample(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    """
    Generate bootstrap samples of accuracy scores.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_samples: Number of bootstrap samples to generate
        
    Returns:
        Array of bootstrap accuracy scores
    """
    n = len(y_true)
    bootstrap_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_scores[i] = accuracy_score(y_true[indices], y_pred[indices])
    
    return bootstrap_scores


def statistical_significance_test(
    model_results: List[Dict[str, Any]], 
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    output_dir: str = "models"
) -> Tuple[pd.DataFrame, str, str]:
    """
    Perform statistical significance testing between models using bootstrap sampling.
    
    Args:
        model_results: List of dictionaries containing model results
                      Each dict should have keys: 'name', 'y_true', 'y_pred'
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        output_dir: Directory to save results
        
    Returns:
        DataFrame with p-values, path to CSV file, and path to plot
    """
    n_models = len(model_results)
    model_names = [result['name'] for result in model_results]
    
    # Calculate bootstrap distributions for each model
    bootstrap_distributions = []
    for result in model_results:
        y_true = np.array(result['y_true'])
        y_pred = np.array(result['y_pred'])
        bootstrap_scores = bootstrap_sample(y_true, y_pred, n_bootstrap)
        bootstrap_distributions.append(bootstrap_scores)
    
    # Calculate p-values for all pairs of models
    p_values = np.zeros((n_models, n_models))
    confidence_intervals = []
    
    for i in range(n_models):
        # Calculate 95% confidence interval for each model
        ci_low = np.percentile(bootstrap_distributions[i], 2.5)
        ci_high = np.percentile(bootstrap_distributions[i], 97.5)
        mean_acc = np.mean(bootstrap_distributions[i])
        confidence_intervals.append({
            'model': model_names[i],
            'mean_accuracy': mean_acc,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
        
        for j in range(n_models):
            if i == j:
                p_values[i, j] = 1.0
                continue
                
            # Calculate p-value using bootstrap distributions
            # Null hypothesis: model i is not better than model j
            diff_distribution = bootstrap_distributions[i] - bootstrap_distributions[j]
            p_value = np.mean(diff_distribution <= 0)
            p_values[i, j] = p_value
    
    # Create DataFrame for p-values
    p_value_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
    
    # Create DataFrame for confidence intervals
    ci_df = pd.DataFrame(confidence_intervals)
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    p_value_csv_path = os.path.join(output_dir, "model_pvalue_comparison.csv")
    p_value_df.to_csv(p_value_csv_path)
    
    ci_csv_path = os.path.join(output_dir, "model_confidence_intervals.csv")
    ci_df.to_csv(ci_csv_path, index=False)
    
    # Create visualization of confidence intervals
    plt.figure(figsize=(12, 6))
    
    # Sort models by mean accuracy
    ci_df = ci_df.sort_values('mean_accuracy', ascending=False)
    
    # Plot confidence intervals
    plt.errorbar(
        x=range(len(ci_df)), 
        y=ci_df['mean_accuracy'], 
        yerr=[(ci_df['mean_accuracy'] - ci_df['ci_low']), (ci_df['ci_high'] - ci_df['mean_accuracy'])],
        fmt='o', 
        capsize=5, 
        elinewidth=2, 
        markersize=8
    )
    
    # Add model names to x-axis
    plt.xticks(range(len(ci_df)), ci_df['model'], rotation=45, ha='right')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy with 95% Confidence Intervals')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_confidence_intervals")
    png_path, svg_path = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    # Create heatmap of p-values
    plt.figure(figsize=(10, 8))
    plt.imshow(p_values, cmap='coolwarm', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            color = 'white' if p_values[i, j] < 0.5 else 'black'
            plt.text(j, i, f'{p_values[i, j]:.3f}', ha='center', va='center', color=color)
            
            # Highlight significant differences
            if i != j and p_values[i, j] < alpha:
                plt.text(j, i, '*', ha='right', va='top', color='white', fontsize=16)
    
    # Add labels and title
    plt.xticks(range(n_models), model_names, rotation=45, ha='right')
    plt.yticks(range(n_models), model_names)
    plt.xlabel('Model B')
    plt.ylabel('Model A')
    plt.title('P-values for H0: Model A is not better than Model B')
    plt.colorbar(label='p-value')
    plt.tight_layout()
    
    # Save figure
    base_filename = os.path.join(output_dir, "model_pvalue_heatmap")
    p_heatmap_png, p_heatmap_svg = save_figure(plt, base_filename, formats=['png', 'svg'])
    
    return p_value_df, p_value_csv_path, png_path


def mcnemar_test(
    model_results: List[Dict[str, Any]],
    alpha: float = 0.05,
    output_dir: str = "models"
) -> Tuple[pd.DataFrame, str]:
    """
    Perform McNemar's test for comparing model predictions.
    
    Args:
        model_results: List of dictionaries containing model results
                      Each dict should have keys: 'name', 'y_true', 'y_pred'
        alpha: Significance level
        output_dir: Directory to save results
        
    Returns:
        DataFrame with test statistics and p-values, path to CSV file
    """
    n_models = len(model_results)
    model_names = [result['name'] for result in model_results]
    
    # Initialize results DataFrame
    results = []
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Get predictions
            y_true = np.array(model_results[i]['y_true'])  # Both models should have same y_true
            y_pred_i = np.array(model_results[i]['y_pred'])
            y_pred_j = np.array(model_results[j]['y_pred'])
            
            # Create contingency table
            # b: both correct, c: model i correct & model j wrong
            # d: model i wrong & model j correct, a: both wrong
            b = np.sum((y_pred_i == y_true) & (y_pred_j == y_true))
            c = np.sum((y_pred_i == y_true) & (y_pred_j != y_true))
            d = np.sum((y_pred_i != y_true) & (y_pred_j == y_true))
            a = np.sum((y_pred_i != y_true) & (y_pred_j != y_true))
            
            # Apply McNemar's test with continuity correction
            if c + d > 0:
                statistic = ((abs(c - d) - 1) ** 2) / (c + d)
                p_value = stats.chi2.sf(statistic, 1)
            else:
                statistic = 0
                p_value = 1.0
                
            # Determine which model is better
            better_model = model_names[i] if c > d else model_names[j] if d > c else "Equal"
            
            # Add to results
            results.append({
                'model_a': model_names[i],
                'model_b': model_names[j],
                'both_correct': b,
                'a_correct_b_wrong': c,
                'a_wrong_b_correct': d,
                'both_wrong': a,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha,
                'better_model': better_model if p_value < alpha else "No significant difference"
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "mcnemar_test_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    return results_df, csv_path


def paired_t_test(
    model_results: List[Dict[str, Any]],
    alpha: float = 0.05,
    output_dir: str = "models"
) -> Tuple[pd.DataFrame, str]:
    """
    Perform paired t-test for comparing model predictions using k-fold cross-validation results.
    
    Args:
        model_results: List of dictionaries containing model results
                      Each dict should have keys: 'name', 'fold_accuracies'
        alpha: Significance level
        output_dir: Directory to save results
        
    Returns:
        DataFrame with test statistics and p-values, path to CSV file
    """
    n_models = len(model_results)
    model_names = [result['name'] for result in model_results]
    
    # Initialize results DataFrame
    results = []
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Get fold accuracies
            acc_i = np.array(model_results[i]['fold_accuracies'])
            acc_j = np.array(model_results[j]['fold_accuracies'])
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(acc_i, acc_j)
            
            # Determine which model is better
            better_model = model_names[i] if np.mean(acc_i) > np.mean(acc_j) else model_names[j]
            
            # Add to results
            results.append({
                'model_a': model_names[i],
                'model_b': model_names[j],
                'mean_diff': np.mean(acc_i - acc_j),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'better_model': better_model if p_value < alpha else "No significant difference"
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "paired_ttest_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    return results_df, csv_path


if __name__ == "__main__":
    # Example usage
    print("Statistical testing module loaded successfully.")
    print("Use statistical_significance_test() to compare model performances.")
    print("Use mcnemar_test() for paired comparison of model predictions.")
    print("Use paired_t_test() for comparing cross-validation results.") 