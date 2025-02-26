import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from PIL import Image
from data_loader import CLASS_NAMES

# Mapping from index to class name
IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# Function to visualize predictions with PyTorch
def visualize_predictions_pytorch(model, test_loader, device, num_images=5):
    """
    Visualize model predictions on test images
    
    Args:
        model: PyTorch model
        test_loader: PyTorch DataLoader for test data
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return
                
                images_so_far += 1
                ax = plt.subplot(num_images // 2 + 1, 2, images_so_far)
                ax.axis('off')
                
                # Convert tensor to numpy for visualization
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Plot image
                ax.imshow(img)
                
                # Get class names
                true_class = IDX_TO_CLASS[labels[j].item()]
                pred_class = IDX_TO_CLASS[preds[j].item()]
                
                # Format class names for display
                true_class = ' '.join(word.capitalize() for word in true_class.replace('_', ' ').split())
                pred_class = ' '.join(word.capitalize() for word in pred_class.replace('_', ' ').split())
                
                # Set title with prediction and ground truth
                title = f"Pred: {pred_class}\nTrue: {true_class}"
                ax.set_title(title, color=("green" if preds[j] == labels[j] else "red"))
    
    plt.tight_layout()
    plt.savefig('models/predictions_pytorch.png')
    plt.show()

# Function to visualize predictions with TensorFlow
def visualize_predictions_tensorflow(model, test_ds, num_images=5):
    """
    Visualize model predictions on test images
    
    Args:
        model: TensorFlow model
        test_ds: TensorFlow Dataset for test data
        num_images: Number of images to visualize
    """
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        preds = np.argmax(predictions, axis=1)
        
        for j in range(len(images)):
            if images_so_far >= num_images:
                return
            
            images_so_far += 1
            ax = plt.subplot(num_images // 2 + 1, 2, images_so_far)
            ax.axis('off')
            
            # Get image
            img = images[j].numpy()
            
            # Reverse preprocessing for visualization
            img = (img * 0.5) + 0.5  # Reverse MobileNetV2 preprocessing
            img = np.clip(img, 0, 1)
            
            # Plot image
            ax.imshow(img)
            
            # Get class names
            true_class = IDX_TO_CLASS[labels[j].numpy()]
            pred_class = IDX_TO_CLASS[preds[j]]
            
            # Format class names for display
            true_class = ' '.join(word.capitalize() for word in true_class.replace('_', ' ').split())
            pred_class = ' '.join(word.capitalize() for word in pred_class.replace('_', ' ').split())
            
            # Set title with prediction and ground truth
            title = f"Pred: {pred_class}\nTrue: {true_class}"
            ax.set_title(title, color=("green" if preds[j] == labels[j] else "red"))
    
    plt.tight_layout()
    plt.savefig('models/predictions_tensorflow.png')
    plt.show()

# Function to predict a single image with PyTorch
def predict_image_pytorch(model, image_path, device, transform=None):
    """
    Predict class for a single image using PyTorch model
    
    Args:
        model: PyTorch model
        image_path: Path to the image
        device: Device to run inference on
        transform: Transforms to apply to the image
        
    Returns:
        Predicted class name and probability
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    if transform:
        image_tensor = transform(image).unsqueeze(0).to(device)
    else:
        # Default transform if none provided
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    # Get class name
    pred_class = IDX_TO_CLASS[pred_idx.item()]
    pred_class = ' '.join(word.capitalize() for word in pred_class.replace('_', ' ').split())
    
    return pred_class, conf.item()

# Function to predict a single image with TensorFlow
def predict_image_tensorflow(model, image_path, img_size=128):
    """
    Predict class for a single image using TensorFlow model
    
    Args:
        model: TensorFlow model
        image_path: Path to the image
        img_size: Size to resize the image to
        
    Returns:
        Predicted class name and probability
    """
    # Load and preprocess image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    
    # Get prediction
    predictions = model.predict(img)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx]
    
    # Get class name
    pred_class = IDX_TO_CLASS[pred_idx]
    pred_class = ' '.join(word.capitalize() for word in pred_class.replace('_', ' ').split())
    
    return pred_class, confidence

# Function to compare model sizes
def compare_model_sizes(pytorch_model, tensorflow_model):
    """
    Compare the sizes of PyTorch and TensorFlow models
    
    Args:
        pytorch_model: PyTorch model
        tensorflow_model: TensorFlow model
    """
    # Calculate PyTorch model size
    param_size = 0
    for param in pytorch_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in pytorch_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    pytorch_size_mb = (param_size + buffer_size) / 1024**2
    
    # Calculate TensorFlow model size
    tf_model_bytes = 0
    for weight in tensorflow_model.weights:
        tf_model_bytes += weight.numpy().nbytes
    
    tensorflow_size_mb = tf_model_bytes / (1024 * 1024)
    
    # Print comparison
    print(f"PyTorch model size: {pytorch_size_mb:.2f} MB")
    print(f"TensorFlow model size: {tensorflow_size_mb:.2f} MB")
    
    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.bar(['PyTorch', 'TensorFlow'], [pytorch_size_mb, tensorflow_size_mb])
    plt.title('Model Size Comparison')
    plt.ylabel('Size (MB)')
    plt.savefig('models/model_size_comparison.png')
    plt.show() 