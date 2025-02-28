import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from cell1_imports_and_constants import IMG_SIZE, BATCH_SIZE, CLASS_NAMES, IDX_TO_CLASS

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

# Data loading function
def load_data(data_dir='dataset', img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.0):
    # Define transformations with more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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
    
    # Print dataset statistics
    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Create validation set if needed
    val_images = []
    val_labels = []
    
    if val_split > 0:
        # Split training data into train and validation sets
        train_indices, val_indices = train_test_split(
            range(len(train_images)), 
            test_size=val_split, 
            stratify=train_labels,
            random_state=42
        )
        
        val_images = [train_images[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        train_images = [train_images[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        
        print(f"After splitting - Training images: {len(train_images)}, Validation images: {len(val_images)}")
    else:
        # Use test set as validation set
        print("Using test set as validation set")
        val_images = test_images
        val_labels = test_labels
    
    # Class distribution
    train_class_counts = {}
    for label in train_labels:
        class_name = IDX_TO_CLASS[label]
        train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1
    
    print("\nClass distribution in training set:")
    for class_name, count in train_class_counts.items():
        print(f"{class_name}: {count} images")
    
    # Create datasets
    train_dataset = BananaLeafDataset(train_images, train_labels, transform=train_transform)
    val_dataset = BananaLeafDataset(val_images, val_labels, transform=val_transform)
    test_dataset = BananaLeafDataset(test_images, test_labels, transform=test_transform)
    
    # Create data loaders with appropriate number of workers
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    return train_loader, val_loader, test_loader 