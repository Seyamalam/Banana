## 1. Introduction

### 1.1 Background and Motivation

Banana (Musa spp.) cultivation represents one of the world's most significant agricultural sectors, serving as both a critical food security crop and an economic cornerstone for many developing regions. With global production exceeding 116 million tonnes annually across over 130 countries, bananas rank as the fourth most important food crop after rice, wheat, and maize in terms of economic value. However, the sustainability of banana production faces considerable threats from various diseases, which can reduce yields by 30-100% if left undetected or mismanaged.

Disease diagnosis in banana cultivation traditionally relies on expert visual inspection of leaf symptoms—a method constrained by the limited availability of agricultural specialists, especially in remote farming communities. The symptoms of major banana diseases including Black Sigatoka (Mycosphaerella fijiensis), Yellow Sigatoka (Mycosphaerella musicola), Panama Disease (Fusarium wilt), and Banana Bunchy Top Virus (BBTV) manifest as characteristic patterns on leaf surfaces, making them potentially identifiable through image analysis. Early detection is particularly crucial, as many banana pathogens become increasingly difficult to control as the infection progresses.

The application of deep learning techniques, particularly Convolutional Neural Networks (CNNs), has emerged as a promising approach to automate plant disease diagnosis. Recent advances in computer vision have demonstrated exceptional accuracy in classifying various crop diseases from digital images. However, significant challenges remain in translating these laboratory achievements into practical agricultural tools. Real-world deployment introduces considerations beyond simple classification accuracy, including:

1. **Environmental Variability**: Field conditions present diverse lighting, angles, backgrounds, and image qualities that can substantially degrade model performance.
   
2. **Resource Constraints**: Agricultural technology, particularly in developing regions, operates under significant computational, power, and connectivity limitations.

3. **Deployment Barriers**: Practical implementation requires consideration of inference speed, model size, memory usage, and compatibility with various hardware platforms.

These challenges highlight the need for a more comprehensive evaluation framework that considers not only ideal-case accuracy but also robustness under variable conditions and performance within computational constraints typical of agricultural settings.

### 1.2 Research Gap and Objectives

While numerous studies have explored CNN applications for plant disease classification, including banana leaf diseases, several critical research gaps remain:

1. Most studies prioritize classification accuracy under controlled conditions, with limited attention to model robustness against environmental perturbations that simulate field deployments.

2. Comparisons between architectures often focus on standard metrics (accuracy, precision, recall) without evaluating deployment-critical factors such as parameter efficiency, memory usage, and inference latency.

3. The trade-offs between custom architectures designed specifically for agricultural applications versus pre-trained general-purpose models remain insufficiently explored, particularly regarding robustness and resource efficiency.

4. Few studies offer concrete, evidence-based guidelines for model selection based on specific deployment scenarios and resource constraints.

To address these gaps, our research aims to provide a systematic, multi-faceted evaluation of CNN models for banana leaf disease classification with the following specific objectives:

1. Implement and compare a custom CNN architecture (BananaLeafCNN) against established models (ResNet50, VGG16, DenseNet121, MobileNetV3, EfficientNetB3) to evaluate trade-offs between model complexity and performance.

2. Assess model robustness through systematic perturbation analysis that simulates various field conditions, including lighting variations, blur, noise, geometric transformations, occlusion, and compression artifacts.

3. Analyze deployment metrics including parameter counts, memory footprints, inference latency across batch sizes, and platform-specific performance characteristics.

4. Develop a framework for model selection based on specific agricultural deployment scenarios, balancing performance requirements with resource constraints.

### 1.3 Scope and Structure

This study focuses on the classification of four major banana leaf disease categories plus healthy leaves, using a dataset of high-quality images collected from various banana-growing regions. Our methodology encompasses model training, validation, robustness testing, and deployment metric collection using standardized protocols to enable fair comparisons.

The remainder of this paper is structured as follows:

- **Section 2**: Literature Review - Examines relevant work in plant disease classification, CNN architectures, robustness evaluation, and deployment optimization.
- **Section 3**: Methodology - Details dataset characteristics, model architectures, training procedures, perturbation protocols, and deployment benchmark methods.
- **Section 4**: Classification Performance Results - Presents accuracy, precision, recall, and F1-scores across models.
- **Section 5**: Ablation Study Findings - Analyzes the impact of architectural components and hyperparameters.
- **Section 6**: Model Efficiency Analysis - Evaluates computational requirements and parameter efficiency.
- **Section 7**: Comparative Analysis Results - Presents direct model comparisons across multiple metrics.
  - **Section 7.3**: Robustness Analysis Results - Examines model performance under various perturbations.
  - **Section 7.4**: Deployment Metrics Results - Analyzes inference speed, model size, and platform-specific performance.
- **Section 8**: Discussion - Synthesizes insights regarding architecture selection, robustness implications, and practical deployment considerations.
- **Section 9**: Limitations and Future Work - Acknowledges constraints and suggests research directions.
- **Section 10**: Conclusion - Summarizes key findings and contributions.

Our research contributes to the growing field of AI-enabled agricultural technology by providing both methodological advances for model evaluation and practical insights for implementing banana leaf disease diagnosis systems across diverse computational environments. 