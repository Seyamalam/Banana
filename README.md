# Banana Leaf Disease Classification

This project implements a lightweight and efficient CNN model for classifying banana leaf diseases using both PyTorch and TensorFlow.

## Dataset

The dataset contains images of banana leaves with 7 different classes:
1. Healthy banana leaf
2. Black Sigatoka
3. Yellow Sigatoka
4. Panama Disease
5. Moko Disease
6. Insect Pest
7. Bract Mosaic Virus

## Project Structure

```
.
├── dataset/
│   ├── train/
│   └── test/
├── requirements.txt
├── data_loader.py
├── model_pytorch.py
├── model_tensorflow.py
├── train_pytorch.py
├── train_tensorflow.py
├── utils.py
└── README.md
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the PyTorch model:
```bash
python train_pytorch.py
```

3. Train the TensorFlow model:
```bash
python train_tensorflow.py
```

## Model Architecture

The custom CNN model is designed to be lightweight and efficient while maintaining high accuracy. It uses:
- Depthwise separable convolutions to reduce parameters
- Batch normalization for faster convergence
- Global average pooling to reduce parameters
- Dropout for regularization

## Performance

The model achieves good accuracy while being small in size:
- Model size: < 5MB
- Training time: Fast on CPU
- Accuracy: Competitive with larger models 