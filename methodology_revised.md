# Methodology

## 1. Dataset and Preprocessing

### 1.1 Banana Leaf Disease Dataset

Our study utilized a comprehensive dataset of banana leaf images spanning seven distinct categories: healthy leaves and six disease conditions (Black Sigatoka, Yellow Sigatoka, Panama Disease, Moko Disease, Insect Pest damage, and Bract Mosaic Virus). These represent the most economically significant banana diseases affecting tropical agriculture. The dataset was organized into training (80%) and testing (20%) splits, with stratified sampling to maintain class distribution.

The standardized directory structure enabled efficient data loading and processing:
```
dataset/
├── train/
│   ├── banana_healthy_leaf/
│   ├── black_sigatoka/
│   └── ... (additional disease categories)
└── test/
    ├── banana_healthy_leaf/
    ├── black_sigatoka/
    └── ... (additional disease categories)
```

### 1.2 Preprocessing and Augmentation Pipeline

We developed a consistent preprocessing pipeline to ensure fair model comparison and optimize BananaLeafCNN's performance:

1. **Resolution Standardization**: All images were resized to 224×224 pixels, balancing detail preservation with computational efficiency.

2. **Color Normalization**: RGB values were normalized using ImageNet-derived statistics (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]) to facilitate stable training.

3. **Data Augmentation**: For training, we applied a targeted augmentation strategy addressing agricultural image variability:
   - Random horizontal and vertical flips (simulating leaf orientation changes)
   - Random rotation (±20°, reflecting capture angle variations)
   - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, mimicking lighting differences)
   - Random affine transformations (translations up to 10%, simulating leaf positioning variance)

These augmentations were particularly important for BananaLeafCNN, as our preliminary testing showed that robust augmentation helped the custom architecture generalize effectively despite its efficient parameter count.

## 2. BananaLeafCNN Architecture Design

### 2.1 Architecture Development Philosophy

BananaLeafCNN was designed with three guiding principles:
1. **Agricultural Domain Specificity**: Tailored for the visual characteristics of leaf diseases
2. **Parameter Efficiency**: Minimizing model size for resource-constrained deployment
3. **Robustness Prioritization**: Maintaining performance under variable field conditions

Rather than adapting general-purpose architectures, we built BananaLeafCNN from first principles, focusing on the specific feature extraction requirements for banana leaf disease diagnosis.

### 2.2 BananaLeafCNN Architecture Specification

Our final BananaLeafCNN architecture features a carefully balanced sequential design:

```python
class BananaLeafCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BananaLeafCNN, self).__init__()
        # Initial feature extraction block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Mid-level feature refinement blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Final feature extraction
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

This architecture represents 0.2M parameters in total, with a progressive spatial dimension reduction while maintaining moderate feature channel width. Key architectural decisions include:

1. **Moderate Channel Width**: Starting with 32 channels, expanding to 64 channels in subsequent layers, balancing expressivity with efficiency
2. **Consistent Convolution Sizing**: 3×3 kernels throughout, maintaining receptive field growth while minimizing parameters
3. **Strategic Normalization**: Batch normalization following each convolution to stabilize training and enhance robustness
4. **Compact Classification Head**: Constrained fully connected layers (64 units) with dropout regularization (0.5)

### 2.3 MobileNetV3 as Comparison Baseline

We selected MobileNetV3-Large as our primary comparison point because:
1. It represents the state-of-the-art in mobile-optimized architectures
2. It was specifically designed for resource-constrained deployment
3. It provides a standardized baseline for evaluating BananaLeafCNN's efficiency gains

The MobileNetV3-Large implementation used the standard architecture from torchvision, with the final classification layer modified to match our 7 disease categories. We initialized it with pre-trained ImageNet weights to ensure optimal transfer learning performance.

## 3. Training Methodology

### 3.1 BananaLeafCNN Training Protocol

BananaLeafCNN was trained using a protocol optimized for its architecture:

- **Optimizer**: Adam with β₁=0.9, β₂=0.999
- **Learning Rate Schedule**: Initial lr=0.001 with reduction on plateau (factor=0.1, patience=5)
- **Batch Size**: 32 samples
- **Training Duration**: 30 epochs (with early stopping based on validation loss)
- **Loss Function**: Cross-entropy loss
- **Device**: NVIDIA GeForce RTX 3080 GPU

We found that BananaLeafCNN converged significantly faster than MobileNetV3, typically reaching optimal performance within 30 epochs compared to 50+ epochs for MobileNetV3. This training efficiency represents another advantage of our domain-specific architecture.

### 3.2 Hyperparameter Optimization

For BananaLeafCNN, we conducted systematic hyperparameter optimization through grid search over:

| Hyperparameter | Values Tested | Final Selection |
|----------------|---------------|----------------|
| Initial Filter Count | 16, 32, 64 | 32 |
| Mid-layer Filter Count | 32, 64, 128 | 64 |
| Dropout Rate | 0.3, 0.5, 0.7 | 0.5 |
| Learning Rate | 0.0001, 0.001, 0.01 | 0.001 |
| Batch Size | 16, 32, 64 | 32 |

This optimization process was crucial for identifying the optimal parameter-performance balance that characterizes BananaLeafCNN's efficiency.

## 4. Evaluation Framework

### 4.1 Classification Performance Assessment

We evaluated BananaLeafCNN's classification performance using:

- **Accuracy**: Overall correct predictions percentage
- **Precision, Recall, and F1-Score**: Per-class and macro-averaged
- **Confusion Matrices**: Both raw counts and normalized proportions

For comparative analysis, MobileNetV3's performance metrics were recorded under identical testing conditions.

### 4.2 Ablation Studies

To understand BananaLeafCNN's architectural design choices, we conducted comprehensive ablation studies by systematically modifying key components:

1. **Batch Normalization Impact**: Removing batch normalization layers
2. **Filter Count Variation**: Increasing/decreasing filters in convolutional layers
3. **Activation Function Alternatives**: Replacing ReLU with Leaky ReLU
4. **Depth Modification**: Adding/removing convolutional blocks
5. **Regularization Adjustment**: Varying dropout rates (0.3, 0.5, 0.7)

Each ablation variant was trained and evaluated using identical protocols, enabling direct comparison of component contributions to overall performance.

### 4.3 Robustness Analysis

BananaLeafCNN's resilience to real-world conditions was systematically evaluated through perturbation testing that simulated field capture variations:

- **Brightness Variation**: Factors of 0.5, 0.75, 1.25, 1.5, 2.0
- **Contrast Variation**: Factors of 0.5, 0.75, 1.25, 1.5, 2.0
- **Gaussian Noise**: Standard deviations of 0.05, 0.1, 0.2, 0.3, 0.5
- **Blur**: Gaussian kernel sizes of 3, 5, 7, 9, 11
- **Rotation**: Angles of 5°, 10°, 15°, 30°, 45°, 90°
- **Occlusion**: Square patches of 10, 20, 30, 40, 50 pixels
- **JPEG Compression**: Quality levels of 90, 80, 70, 60, 50, 40, 30, 20, 10

For each perturbation type and level, we measured the relative accuracy drop compared to unperturbed images. MobileNetV3 underwent identical testing to provide comparative robustness profiles.

### 4.4 Deployment Metrics Evaluation

To assess BananaLeafCNN's suitability for resource-constrained environments, we measured:

- **Parameter Count**: Total model parameters
- **Model Size**: Storage requirements in MB
- **Memory Usage**: Peak RAM consumption during inference
- **Inference Latency**: Per-image processing time on various platforms
- **Throughput**: Images processed per second across batch sizes
- **GPU Acceleration Factor**: Speedup when moving from CPU to GPU

For platform testing, we evaluated BananaLeafCNN across multiple environments:
- Desktop CPU (Intel Core i9-10900K)
- Desktop GPU (NVIDIA GeForce RTX 3080)
- Raspberry Pi 4 (representing edge devices)
- Mobile CPU (Snapdragon 855)

We also evaluated export format impact by comparing model performance when exported to:
- PyTorch native format
- ONNX
- TorchScript

Each measurement included appropriate warmup iterations and multiple runs to ensure statistical reliability.

## 5. Implementation Details

### 5.1 Software Framework

BananaLeafCNN was implemented using:
- Python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0
- NumPy 1.21.2
- pandas 1.3.3
- Matplotlib 3.4.3
- PIL 8.3.1

Additional tools for deployment analysis:
- thop 0.0.31 (for FLOPs calculation)
- ONNX Runtime 1.9.0 (for optimized inference)
- PyTorch Mobile (for edge deployment testing)

### 5.2 Reproducibility Measures

To ensure reproducibility of BananaLeafCNN's development and evaluation:
- All random operations used fixed seeds (42)
- Training procedures were documented through logging
- Model checkpoints were saved at regular intervals
- Evaluation protocols were standardized across all tests
- Source code and model weights were versioned using Git

## 6. Experimental Process Flow

Our experimental workflow proceeded as follows:

1. **BananaLeafCNN Architecture Development**
   - Initial design based on domain understanding
   - Iterative refinement through validation testing
   - Final architecture selection

2. **Model Training**
   - BananaLeafCNN training with optimized hyperparameters
   - MobileNetV3 fine-tuning for comparison

3. **Performance Evaluation**
   - Classification metrics
   - Ablation analysis
   - Robustness profiling
   - Deployment metrics

4. **Analysis and Interpretation**
   - Performance-efficiency trade-off analysis
   - Robustness characterization
   - Deployment optimization opportunities
   - Real-world implementation recommendations

This comprehensive methodology enabled us to thoroughly evaluate BananaLeafCNN's capabilities and properly contextualize its advantages for agricultural deployment scenarios. 