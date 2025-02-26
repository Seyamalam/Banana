import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_SIZE = 224  # Increased image size for better feature extraction
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

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

# Inverse mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
def load_data(data_dir='dataset', img_size=IMG_SIZE, batch_size=BATCH_SIZE):
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
    test_dataset = BananaLeafDataset(test_images, test_labels, transform=test_transform)
    
    # Create data loaders with appropriate number of workers
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader

# CNN Model similar to the provided TensorFlow architecture
class BananaLeafCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(BananaLeafCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fifth convolutional block
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Sixth convolutional block
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after all convolutions and pooling
        # For 224x224 input, after 6 pooling layers (each dividing by 2), we get 3x3 feature maps
        self.flatten_size = 64 * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout = nn.Dropout(0.5)  # Added dropout for regularization
        self.fc2 = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Fifth block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        # Sixth block
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool6(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    # Calculate classification report
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    
    return epoch_loss, epoch_acc, report

# Visualize predictions
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(20, 10))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    model.train()  # Reset to train mode
                    return
                
                images_so_far += 1
                ax = plt.subplot(2, num_images//2, images_so_far)
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
    plt.savefig('predictions.png')
    plt.show()
    model.train()  # Reset to train mode

# Plot confusion matrix
def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Computing confusion matrix"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    class_names = [' '.join(word.capitalize() for word in IDX_TO_CLASS[i].replace('_', ' ').split()) 
                  for i in range(NUM_CLASSES)]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    model.train()  # Reset to train mode

# Main function
def main():
    # Set random seed
    set_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize model
    model = BananaLeafCNN(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Print model summary
    num_params = count_parameters(model)
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nNumber of parameters: {num_params:,}")
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc, report = validate(model, test_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("\nClassification Report:")
        print(report)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'models/best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, 'models/final_model.pth')
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(model, test_loader, device)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(model, test_loader, device)

# Entry point
if __name__ == "__main__":
    main() 