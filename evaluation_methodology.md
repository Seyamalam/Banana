# Evaluation Methodology for Banana Leaf Disease Classification

## 1. Introduction to Model Evaluation

Evaluation methodology refers to the systematic approach used to assess model performance in classifying banana leaf diseases. A robust evaluation framework is essential to:

- Accurately measure model performance across multiple disease categories
- Compare different model architectures objectively
- Identify strengths and weaknesses in classification performance
- Ensure reliability for agricultural applications
- Guide model selection for deployment in real-world settings

Our evaluation methodology follows best practices in machine learning assessment, with a specific focus on agricultural disease detection challenges.

## 2. Evaluation Metrics

We employ a comprehensive set of metrics to evaluate model performance, providing a multi-faceted view of classification capability.

### 2.1 Primary Metrics

#### Accuracy
The most fundamental metric, representing the proportion of correctly classified images:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

While valuable for overall assessment, accuracy alone can be misleading in cases of class imbalance.

#### Precision
Measures the model's ability to avoid false positives for each disease class:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

This is crucial for agricultural applications where misdiagnosis can lead to unnecessary treatments.

#### Recall
Quantifies the model's ability to detect all instances of a disease:

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

High recall is vital in agricultural settings to ensure diseased plants are not missed.

#### F1-Score
The harmonic mean of precision and recall, providing a balanced measure:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

This metric is especially useful when seeking a balance between missing diseases and false alarms.

### 2.2 Confusion Matrix Analysis

We generate and analyze confusion matrices to gain deeper insights into model performance:

- **Raw Confusion Matrix**: Counts of predictions for each true class
- **Normalized Confusion Matrix**: Proportions of predictions for each true class
- **Per-class Accuracy**: Diagonal elements of the normalized confusion matrix
- **Error Analysis**: Off-diagonal elements to identify commonly confused disease pairs

Confusion matrices are visualized using heatmaps for intuitive interpretation and saved in both visual formats (PNG, SVG) and data formats (CSV) for further analysis.

### 2.3 Per-Class Metrics

To address potential class imbalance, we calculate precision, recall, and F1-score for each disease category:

- **Banana Healthy Leaf**: Baseline for comparison
- **Black Sigatoka**: Critical fungal disease
- **Yellow Sigatoka**: Less severe fungal disease
- **Panama Disease**: Fusarium wilt
- **Moko Disease**: Bacterial wilt
- **Insect Pest**: Various insect damages
- **Bract Mosaic Virus**: Viral infection

Class-specific metrics provide insights into disease-specific detection performance, revealing whether a model exhibits bias toward particular diseases or environmental conditions.

## 3. Evaluation Process

### 3.1 Test Dataset Evaluation

Our evaluation process follows a systematic approach:

1. **Model Loading**: Load trained model weights from checkpoints
2. **Data Preparation**: Process test data with appropriate transformations
3. **Inference Loop**: 
   - Set model to evaluation mode (`model.eval()`)
   - Disable gradient calculation (`with torch.no_grad()`)
   - Forward pass all test images through the model
   - Record predictions and true labels
4. **Metrics Calculation**: 
   - Compute confusion matrix using scikit-learn
   - Calculate accuracy, precision, recall, and F1-score
   - Generate per-class metrics
5. **Visualization**:
   - Plot confusion matrices
   - Create sample prediction visualizations
   - Generate comparison charts

The evaluation is performed using a completely held-out test set to ensure unbiased assessment of model performance.

### 3.2 Implementation Details

The evaluation process is implemented in the `evaluate_model` function in `cell6_utils.py`:

```python
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Ensure confusion matrix is a numpy array
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Calculate normalized confusion matrix
    if isinstance(cm, np.ndarray) and cm.size > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1)
            cm_norm = np.zeros_like(cm, dtype=float)
            for i, row_sum in enumerate(row_sums):
                if row_sum > 0:
                    cm_norm[i] = cm[i] / row_sum
    else:
        cm_norm = np.array([[0]])
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confusion_matrix_norm': cm_norm
    }, true_labels, predictions
```

### 3.3 Sample Prediction Visualization

To provide qualitative insights, we visualize sample predictions:

1. Select a batch of test images
2. Generate predictions using the model
3. Create a grid visualization showing:
   - Original image
   - True label
   - Predicted label
   - Color-coding (green for correct, red for incorrect)

This visual analysis helps identify patterns in successful and failed predictions, providing insights beyond numerical metrics.

## 4. Model Comparison Framework

### 4.1 Multi-Model Evaluation

Our research employs systematic comparison across multiple model architectures:

1. **Side-by-Side Metrics**: Direct comparison of accuracy, precision, recall, and F1-score
2. **Visual Comparisons**: 
   - Confusion matrix grid visualization
   - Bar charts of performance metrics
   - Visualization of statistical significance

### 4.2 Statistical Significance Testing

We employ rigorous statistical methods to determine if performance differences between models are significant:

#### Bootstrap Confidence Intervals
For each model, we:
1. Create bootstrap samples by randomly sampling with replacement from test predictions
2. Calculate accuracy for each bootstrap sample
3. Compute 95% confidence intervals for model accuracy
4. Visualize confidence intervals to identify overlaps

#### McNemar's Test
For paired comparison of models' predictions:
1. Create contingency tables counting cases where:
   - Both models are correct/incorrect
   - One model is correct while the other is incorrect
2. Calculate McNemar's chi-squared statistic:

   $$\chi^2 = \frac{(|c - d| - 1)^2}{c + d}$$

   Where:
   - c: cases where model A is correct and model B is incorrect
   - d: cases where model A is incorrect and model B is correct

3. Derive p-values to determine if differences are statistically significant

This test is particularly valuable as it directly compares models on the same test examples, providing stronger evidence of performance differences than aggregate metrics alone.

### 4.3 Comprehensive Comparison Output

For each evaluation run, we generate:

1. **CSV Files**:
   - Overall metrics comparison
   - Per-class performance metrics
   - Statistical test results
   - Confusion matrices in tabular format

2. **Visualizations**:
   - Bar charts for each metric
   - Confusion matrix heatmaps
   - Confidence interval plots
   - Statistical significance heatmaps

3. **Sample Predictions**:
   - Grid visualizations of example classifications
   - Highlighting of successful and failed cases

## 5. Integration with Research Pipeline

### 5.1 Command-Line Interface

The evaluation framework is integrated into the main analysis script with specific flags:

```
python run_analysis.py --evaluate --models resnet18 mobilenet_v2
```

Or as part of a comprehensive analysis:

```
python run_analysis.py --all
```

### 5.2 Output Organization

Evaluation results are organized systematically:

1. **Model-Specific Directories**:
   - `models/evaluation/{model_name}/`: Contains model-specific results
   - Confusion matrices, classification reports, and visualizations

2. **Comparison Directory**:
   - `models/comparisons/evaluation/`: Contains cross-model comparisons
   - Statistical test results, comparative visualizations

### 5.3 Connection to Other Analyses

Our evaluation methodology connects directly to other analyses in the research pipeline:

1. **Training**: Uses the same model architectures and data splitting approach
2. **Ablation Studies**: Provides baseline metrics for component analysis
3. **Robustness Testing**: Establishes baseline performance for perturbation analysis
4. **Deployment Metrics**: Balances accuracy metrics against efficiency considerations

## 6. Real-World Application Context

The evaluation methodology is designed specifically for agricultural applications, with considerations for:

| Evaluation Aspect | Agricultural Relevance |
|-------------------|------------------------|
| Per-class metrics | Different diseases have varying economic impacts |
| Precision focus | Avoid unnecessary pesticide application |
| Recall emphasis | Ensure early disease detection |
| F1-score balance | Practical trade-off for field deployment |
| Confusion matrix | Understand common misdiagnosis patterns |

By implementing this comprehensive evaluation methodology, we ensure that our banana leaf disease classification models are rigorously assessed for both statistical performance and practical agricultural applicability. This approach provides confidence in model selection for deployment in real-world settings where accurate disease diagnosis is crucial for crop protection and sustainable banana production. 