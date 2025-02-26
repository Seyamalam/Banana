import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 7

# Class mapping
CLASS_NAMES = {
    'banana_healthy_leaf': 0,
    'black_sigatoka': 1,
    'yellow_sigatoka': 2,
    'panama_disease': 3,
    'moko_disease': 4,
    'insect_pest': 5,
    'bract_mosaic_virus': 6
}

# PyTorch Dataset
class BananaLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data loading functions for PyTorch
def load_data_pytorch(data_dir='dataset', img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect image paths and labels
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = CLASS_NAMES[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    train_images.append(img_path)
                    train_labels.append(class_idx)
    
    # Load test data
    test_dir = os.path.join(data_dir, 'test')
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = CLASS_NAMES[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    test_images.append(img_path)
                    test_labels.append(class_idx)
    
    # Create datasets
    train_dataset = BananaLeafDataset(train_images, train_labels, transform=train_transform)
    test_dataset = BananaLeafDataset(test_images, test_labels, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Data loading functions for TensorFlow
def _parse_function(filename, label, img_size=IMG_SIZE):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def load_data_tensorflow(data_dir='dataset', img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    # Collect image paths and labels
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = CLASS_NAMES[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    train_images.append(img_path)
                    train_labels.append(class_idx)
    
    # Load test data
    test_dir = os.path.join(data_dir, 'test')
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = CLASS_NAMES[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    test_images.append(img_path)
                    test_labels.append(class_idx)
    
    # Convert to tensors
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(lambda x, y: _parse_function(x, y, img_size), 
                           num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(train_images))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(lambda x, y: _parse_function(x, y, img_size),
                         num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

# Get class weights to handle imbalanced data
def get_class_weights(data_dir='dataset'):
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    
    train_dir = os.path.join(data_dir, 'train')
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = CLASS_NAMES[class_name]
            class_counts[class_idx] = len([f for f in os.listdir(class_dir) 
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items() if count > 0}
    
    return class_weights 