## Title Options

**BananaLeafCNN: An Efficient Deep Learning Architecture for Banana Leaf Disease Classification in Resource-Constrained Agricultural Settings**

**Optimizing for Deployment: Development and Evaluation of BananaLeafCNN for Efficient Banana Leaf Disease Diagnosis**

**BananaLeafCNN: Balancing Accuracy, Robustness, and Computational Efficiency for Agricultural Disease Detection**

**From Lab to Field: BananaLeafCNN's Domain-Specific Architecture for Practical Banana Leaf Disease Diagnosis**

**BananaLeafCNN: A Lightweight Architecture Optimized for Real-World Agricultural Deployment**

## Abstract

This study presents the development and comprehensive evaluation of BananaLeafCNN, a custom-designed convolutional neural network specifically tailored for banana leaf disease classification. Banana crops, vital for food security and economic stability in many tropical regions, face significant threats from various diseases that can be identified through leaf symptoms. While existing deep learning approaches using pre-trained architectures have shown promise, their computational demands often limit practical deployment in resource-constrained agricultural settings.

Our research introduces BananaLeafCNN, a highly efficient architecture that achieves 90.9% classification accuracy while requiring only 0.2M parameters—a 20× reduction compared to MobileNetV3 (4.2M parameters) and a 655× reduction compared to VGG16 (134M parameters). We systematically evaluate BananaLeafCNN's performance across multiple dimensions: classification accuracy, robustness to environmental variations, computational efficiency, and deployment metrics. Through comprehensive ablation studies, we identified that architectural components like batch normalization significantly impact model performance, and our experimental results demonstrate that moderate filter counts (64-128) provide optimal feature extraction for leaf disease patterns.

BananaLeafCNN demonstrates a distinctive performance profile across various operational conditions, with particular strengths in efficient inference and deployment metrics. With just 0.8MB storage footprint, 115ms CPU inference latency, and 3.4ms GPU latency, BananaLeafCNN enables deployment on resource-constrained devices common in agricultural settings. Its exceptional 34× GPU acceleration factor further enhances throughput for batch processing scenarios, reaching up to 3,831 samples per second with optimal batch sizing.

Our findings contribute to agricultural computer vision by demonstrating that domain-specific architectural design can dramatically outperform the transfer learning paradigm in efficiency while maintaining competitive accuracy. We establish that beyond a critical parameter threshold (approximately 0.2M for banana leaf disease classification), additional model capacity yields diminishing returns—challenging conventional approaches to agricultural vision tasks. BananaLeafCNN provides a practical solution for implementing banana disease diagnosis systems across diverse computational environments, from basic smartphones to edge devices, particularly benefiting regions with limited connectivity and computational resources.

Keywords: BananaLeafCNN, deep learning, agricultural computer vision, model efficiency, deployment optimization, environmental robustness, edge computing, resource-constrained applications 