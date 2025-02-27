# Banana Leaf Disease Classification

This project implements a custom CNN model for banana leaf disease classification, focusing on efficiency and small file size. The model is designed to be trained on a CPU and provides competitive performance compared to popular pre-trained models.

## Project Structure

The project is organized into modular components:

- **Data Loading**: `cell2_dataset.py` - Handles dataset loading and preprocessing
- **Model Architecture**: `cell3_model.py` - Defines the custom BananaLeafCNN model
- **Training**: `cell4_training.py` - Contains training and validation functions
- **Visualization**: `cell5_visualization.py` - Provides visualization utilities
- **Utilities**: `cell6_utils.py` - Contains utility functions for model evaluation
- **Model Zoo**: `cell8_model_zoo.py` - Provides implementations of popular models for comparison
- **Analysis Modules**:
  - `cell11_training_resources.py` - Measures training time and resource consumption
  - `cell12_statistical_testing.py` - Performs statistical significance testing
  - `cell13_efficiency_metrics.py` - Calculates efficiency metrics
  - `cell14_ablation_studies.py` - Implements ablation studies
  - `cell15_flops_analysis.py` - Analyzes computational complexity
  - `cell16_robustness_testing.py` - Tests model robustness to perturbations
  - `cell17_cross_dataset.py` - Evaluates cross-dataset performance
  - `cell18_deployment_metrics.py` - Measures deployment metrics

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── train/
│   ├── bacterial_wilt/
│   ├── black_sigatoka/
│   ├── healthy/
│   ├── panama_disease/
│   ├── pestalotiopsis/
│   └── yellow_sigatoka/
└── test/
    ├── bacterial_wilt/
    ├── black_sigatoka/
    ├── healthy/
    ├── panama_disease/
    ├── pestalotiopsis/
    └── yellow_sigatoka/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/banana-leaf-classification.git
cd banana-leaf-classification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

The project provides a main script `run_analysis.py` that can run different analyses:

### Training Models

To train the custom model:
```bash
python run_analysis.py --train --models banana_leaf_cnn
```

To train multiple models for comparison:
```bash
python run_analysis.py --train --models banana_leaf_cnn mobilenet_v2 resnet18
```

### Evaluating Models

To evaluate pre-trained models:
```bash
python run_analysis.py --evaluate --models banana_leaf_cnn mobilenet_v2 resnet18
```

### Running Ablation Studies

To run ablation studies on model components:
```bash
python run_analysis.py --ablation --models banana_leaf_cnn
```

### Testing Robustness

To test model robustness to various perturbations:
```bash
python run_analysis.py --robustness --models banana_leaf_cnn
```

### Measuring Deployment Metrics

To measure deployment metrics (latency, throughput, export formats):
```bash
python run_analysis.py --deployment --models banana_leaf_cnn
```

### Running All Analyses

To run all analyses:
```bash
python run_analysis.py --all --models banana_leaf_cnn mobilenet_v2 resnet18
```

## Command-line Arguments

- `--data_dir`: Path to dataset directory (default: 'dataset')
- `--output_dir`: Path to output directory (default: 'models')
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 30)
- `--lr`: Learning rate (default: 0.001)
- `--seed`: Random seed (default: 42)
- `--train`: Train models
- `--evaluate`: Evaluate models
- `--ablation`: Run ablation studies
- `--robustness`: Run robustness tests
- `--deployment`: Run deployment metrics
- `--all`: Run all analyses
- `--models`: Models to analyze (choices: banana_leaf_cnn, mobilenet_v2, efficientnet_b0, resnet18, shufflenet_v2)

## Results

The analysis results are saved in the output directory (default: 'models') with the following structure:

```
models/
├── banana_leaf_cnn_best.pt          # Best model checkpoint
├── banana_leaf_cnn_final.pt         # Final model checkpoint
├── model_pvalue_comparison.csv      # Statistical significance results
├── ablation/                        # Ablation study results
├── robustness/                      # Robustness test results
├── deployment/                      # Deployment metrics
└── efficiency/                      # Efficiency metrics
```

## Model Architecture

The custom BananaLeafCNN model is designed to be lightweight and efficient while maintaining competitive accuracy. It consists of:

- 6 convolutional blocks with increasing channel dimensions
- Each block contains Conv2D, BatchNorm, ReLU, and MaxPooling layers
- A fully connected layer with 64 neurons before the final classification layer
- Dropout regularization with rate 0.5

## License

This project is licensed under the MIT License - see the LICENSE file for details. 