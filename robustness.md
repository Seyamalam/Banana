# Robustness in Banana Leaf Disease Classification

## 1. Introduction to Model Robustness

Robustness in machine learning refers to a model's ability to maintain performance when faced with variations, perturbations, or adversarial examples in the input data. For deep learning models deployed in agricultural applications, robustness is particularly critical as these systems must operate reliably in uncontrolled environments where lighting conditions, image quality, viewpoints, and other factors can vary significantly from the training data.

In the context of banana leaf disease classification, a robust model should correctly identify diseases regardless of:
- Different lighting conditions (morning, noon, evening, cloudy, sunny)
- Image quality variations (smartphone cameras with different resolutions)
- Viewpoint changes (angles, distances, orientations)
- Partial occlusions (overlapping leaves, insect presence, water droplets)
- Image compression artifacts (common in transmitted or stored images)

## 2. Importance of Robustness for Agricultural Applications

Robustness testing is essential for our banana leaf disease classification system for several reasons:

### 2.1 Real-world Deployment Challenges

Agricultural environments present unique challenges:
- **Variable Field Conditions**: Unlike controlled laboratory settings, field conditions vary dramatically with weather, time of day, and season
- **Device Heterogeneity**: Images may be captured using different devices with varying capabilities
- **User Expertise Variation**: Images may be taken by users with different levels of photography expertise
- **Connectivity Limitations**: In remote agricultural areas, images may undergo compression for transmission over limited bandwidth networks

### 2.2 Economic Implications

The consequences of misclassification in agricultural disease detection can be severe:
- False negatives may lead to unchecked disease spread and crop loss
- False positives may result in unnecessary and costly treatment applications
- Erroneous diagnosis could lead to inappropriate interventions that waste resources

### 2.3 Adoption and Trust

For technological solutions to be adopted by farmers and agricultural extension workers:
- Models must perform consistently under diverse conditions
- Systems must be reliable even when images are not captured under ideal circumstances
- Users must develop trust in the system through consistent performance

## 3. Robustness Evaluation Framework

Our research employs a comprehensive framework to systematically evaluate model robustness through controlled perturbation testing.

### 3.1 General Methodology

The robustness evaluation framework follows these key steps:

1. **Baseline Establishment**: Measure model performance on clean, unperturbed test data
2. **Perturbation Application**: Apply controlled perturbations of increasing intensity to test images
3. **Performance Measurement**: Evaluate model performance on perturbed images
4. **Robustness Profiling**: Plot performance metrics against perturbation intensity
5. **Cross-Model Comparison**: Compare robustness profiles across different model architectures

### 3.2 Perturbation Types and Mathematical Formulations

We test seven distinct perturbation types that simulate real-world conditions:

#### 3.2.1 Gaussian Noise

Gaussian noise simulates sensor noise from cameras, particularly in low-light conditions.

**Mathematical Formulation:**
For an image $I$ with pixel values normalized to [0,1], the noisy image $I'$ is:

$$I'(x,y) = \text{clip}_{[0,1]}(I(x,y) + \mathcal{N}(0, \sigma^2))$$

Where:
- $\mathcal{N}(0, \sigma^2)$ is Gaussian noise with mean 0 and variance $\sigma^2$
- $\text{clip}_{[0,1]}$ ensures values remain in the valid range

We test at $\sigma \in \{0.05, 0.1, 0.2, 0.3, 0.5\}$.

#### 3.2.2 Gaussian Blur

Blur simulates focus issues, motion blur, or images taken in poor conditions.

**Mathematical Formulation:**
For an image $I$, the blurred image $I'$ is:

$$I'(x,y) = \sum_{i=-k}^{k}\sum_{j=-k}^{k} G(i,j) \cdot I(x+i,y+j)$$

Where:
- $G$ is a Gaussian kernel of size $(2k+1) \times (2k+1)$
- $G(i,j) = \frac{1}{2\pi\sigma^2}e^{-\frac{i^2+j^2}{2\sigma^2}}$

We test with kernel sizes $\in \{3, 5, 7, 9, 11\}$.

#### 3.2.3 Brightness Variation

Brightness variations simulate different lighting conditions or exposure settings.

**Mathematical Formulation:**
For an image $I$, the brightness-adjusted image $I'$ is:

$$I'(x,y) = \text{clip}_{[0,1]}(b \cdot I(x,y))$$

Where:
- $b$ is the brightness factor
- $b < 1$ darkens the image
- $b > 1$ brightens the image

We test at $b \in \{0.5, 0.75, 1.25, 1.5, 2.0\}$.

#### 3.2.4 Contrast Variation

Contrast variations simulate different camera settings or lighting conditions affecting image contrast.

**Mathematical Formulation:**
For an image $I$, the contrast-adjusted image $I'$ is:

$$I'(x,y) = \text{clip}_{[0,1]}(c \cdot (I(x,y) - 0.5) + 0.5)$$

Where:
- $c$ is the contrast factor
- $c < 1$ reduces contrast
- $c > 1$ increases contrast

We test at $c \in \{0.5, 0.75, 1.25, 1.5, 2.0\}$.

#### 3.2.5 Rotation

Rotation simulates different viewpoints or image orientations.

**Mathematical Formulation:**
For an image $I$, the rotated image $I'$ is:

$$I'(x',y') = I(x\cos\theta - y\sin\theta, x\sin\theta + y\cos\theta)$$

Where:
- $\theta$ is the rotation angle in degrees
- Bilinear interpolation is used for non-integer coordinates

We test at $\theta \in \{5°, 10°, 15°, 30°, 45°, 90°\}$.

#### 3.2.6 Occlusion

Occlusion simulates partially obscured leaves due to overlapping, insect presence, or other obstructions.

**Implementation:**
For an image $I$, a square region of size $s \times s$ is replaced with black pixels (zero values) at a random location.

We test with occlusion sizes $s \in \{10, 20, 30, 40, 50\}$ pixels.

#### 3.2.7 JPEG Compression

JPEG compression simulates artifacts from image storage or transmission, especially relevant in bandwidth-limited rural areas.

**Implementation:**
Images are saved as JPEG files with varying quality factors and then reloaded.

We test with quality levels $q \in \{90, 80, 70, 60, 50, 40, 30, 20, 10\}$.

### 3.3 Robustness Metrics

For each perturbation type and intensity level, we compute:

1. **Accuracy**: The percentage of correctly classified images
2. **Precision**: The weighted precision across all classes
3. **Recall**: The weighted recall across all classes
4. **F1-Score**: The weighted harmonic mean of precision and recall

Additionally, we calculate derived metrics for comparative analysis:

1. **Accuracy Drop**: The absolute difference between baseline accuracy and accuracy under perturbation
2. **Relative Accuracy Drop**: The percentage decrease in accuracy relative to the baseline performance

## 4. Implementation Details

### 4.1 Perturbation Generation

Perturbations are implemented in our codebase using the following techniques:

- **Gaussian Noise**: Applied by adding random normal noise to tensor values and clipping to valid range
- **Blur**: Implemented using PIL's `ImageFilter.GaussianBlur` with varying radius parameters
- **Brightness and Contrast**: Implemented using PIL's `ImageEnhance.Brightness` and `ImageEnhance.Contrast` with varying enhancement factors
- **Rotation**: Applied using PIL's `Image.rotate` method with different angle values
- **Occlusion**: Implemented by setting rectangular patches of the image tensor to zero at random locations
- **JPEG Compression**: Applied by saving images to a BytesIO buffer with different quality settings and reloading them

### 4.2 Evaluation Process

Our robustness evaluation process is implemented in the `RobustnessTest` class with the following workflow:

1. Initialize with a trained model and test dataset
2. Evaluate baseline performance on clean, unperturbed data
3. For each perturbation type:
   a. Apply perturbations at increasing intensity levels
   b. Evaluate model performance at each level
   c. Store and analyze results
4. Generate visualizations showing performance degradation curves
5. Create summary reports comparing robustness across perturbation types

The implementation supports both:
- Testing a single model against multiple perturbation types
- Comparing multiple models against a specific perturbation type

### 4.3 Controlled Variables

To ensure fair comparison across models, our implementation maintains consistent:
- Test dataset (same images for all models)
- Perturbation parameters (identical intensity levels)
- Evaluation metrics (consistent calculation methodology)
- Random seeds (for reproducible occlusion positions)

## 5. Connection to Real-world Scenarios

Each perturbation type is directly connected to real-world scenarios in agricultural applications:

| Perturbation Type | Real-world Scenario |
|-------------------|---------------------|
| Gaussian Noise | Images taken in low light or with low-quality cameras |
| Blur | Out-of-focus images, hand movement during capture, rain/moisture on lens |
| Brightness Variation | Photos taken at different times of day, under shade vs. direct sunlight |
| Contrast Variation | Different camera settings, overcast vs. sunny conditions |
| Rotation | Different angles of image capture, leaf orientation variability |
| Occlusion | Overlapping leaves, insect presence, debris, water droplets |
| JPEG Compression | Images shared via messaging apps, email, or limited bandwidth connections |

## 6. Expected Outcomes

The robustness analysis will provide:

1. **Quantitative Measurements**: Precise measurements of how performance degrades under various perturbations
2. **Comparative Analysis**: Objective comparisons of different model architectures' robustness characteristics
3. **Vulnerability Identification**: Specific perturbation types that most significantly impact each model
4. **Design Insights**: Guidelines for improving model architecture to enhance robustness

By identifying which models maintain accuracy under challenging conditions, this analysis will help select architectures that not only perform well in controlled environments but remain effective when deployed in real agricultural settings. 