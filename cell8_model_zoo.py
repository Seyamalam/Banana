import torch
import torch.nn as nn
from torchvision.models import get_model, get_model_weights
import torchvision.transforms as transforms
from cell1_imports_and_constants import IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES

def get_available_classification_models():
    """
    Returns a list of available classification models from torchvision
    that can be used for our task.
    """
    # List of models we want to use for comparison
    models = [
        "resnet18",
        "resnet50",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b3",
        "convnext_tiny",
        "densenet121",
        "vgg16",
        "swin_t"
    ]
    
    return models

def create_model_adapter(model_name, pretrained=True, freeze_backbone=False):
    """
    Creates a model adapter that wraps a pre-trained torchvision model
    and adapts it for our banana leaf classification task.
    
    Args:
        model_name: Name of the torchvision model to use
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze the backbone parameters
    
    Returns:
        model: Adapted model
        input_transforms: Transforms required by the model
    """
    # Get the pre-trained weights if specified
    weights = "DEFAULT" if pretrained else None
    
    # Load the base model
    try:
        base_model = get_model(model_name, weights=weights)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None
    
    # Get input transforms
    if pretrained and weights == "DEFAULT":
        # Get the weights object to get the transforms
        weights_enum = get_model_weights(model_name)
        if weights_enum is not None:
            default_weights = weights_enum.DEFAULT
            # Get the transforms from the weights
            input_transforms = default_weights.transforms()
        else:
            # Fallback to standard transforms
            input_transforms = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        # Standard transforms for non-pretrained models
        input_transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Modify the last layer based on model architecture
    if hasattr(base_model, 'fc'):  # ResNet, EfficientNet
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Linear):  # MobileNet, ConvNeXt
        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Sequential):  # VGG, DenseNet
        if hasattr(base_model.classifier, 'out_features'):
            in_features = base_model.classifier.out_features
            base_model.classifier = nn.Linear(in_features, NUM_CLASSES)
        else:
            # For VGG-like networks
            base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, NUM_CLASSES)
    elif hasattr(base_model, 'head'):  # Swin Transformer
        in_features = base_model.head.in_features
        base_model.head = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model architecture for {model_name}")
    
    # Freeze backbone if specified
    if freeze_backbone:
        for name, param in base_model.named_parameters():
            if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
    
    # Create a wrapper class to handle any model-specific needs
    class ModelAdapter(nn.Module):
        def __init__(self, model, model_name):
            super().__init__()
            self.model = model
            self.model_name = model_name
            
        def forward(self, x):
            return self.model(x)
        
        def __str__(self):
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return f"ModelAdapter({self.model_name}) - Parameters: {total_params:,} (trainable: {trainable_params:,})"
    
    # Wrap the model
    model_adapter = ModelAdapter(base_model, model_name)
    
    return model_adapter, input_transforms

class ModelInfo:
    """
    Class to store model information for comparison
    """
    def __init__(self, model_name, model_type, params, trainable_params, model_size_mb, 
                 accuracy=None, loss=None, inference_time=None, class_accuracies=None):
        self.model_name = model_name
        self.model_type = model_type  # 'custom' or 'pretrained'
        self.params = params
        self.trainable_params = trainable_params
        self.model_size_mb = model_size_mb
        self.accuracy = accuracy
        self.loss = loss
        self.inference_time = inference_time
        self.class_accuracies = class_accuracies
        
    def to_dict(self):
        """Convert ModelInfo to dictionary for easy export"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'params': self.params,
            'trainable_params': self.trainable_params,
            'model_size_mb': self.model_size_mb,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'inference_time': self.inference_time,
            'class_accuracies': self.class_accuracies
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create ModelInfo from dictionary"""
        return cls(
            model_name=data['model_name'],
            model_type=data['model_type'],
            params=data['params'],
            trainable_params=data['trainable_params'],
            model_size_mb=data['model_size_mb'],
            accuracy=data['accuracy'],
            loss=data['loss'],
            inference_time=data['inference_time'],
            class_accuracies=data['class_accuracies']
        )

def get_model_info_from_model(model, model_type='custom'):
    """Extract model information from a PyTorch model"""
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    size_all_mb = 0
    for param in model.parameters():
        size_all_mb += param.nelement() * param.element_size()
    for buffer in model.buffers():
        size_all_mb += buffer.nelement() * buffer.element_size()
    size_all_mb = size_all_mb / (1024 * 1024)
    
    # Get model name
    if hasattr(model, 'model_name'):
        model_name = model.model_name
    else:
        model_name = model.__class__.__name__
    
    return ModelInfo(
        model_name=model_name,
        model_type=model_type,
        params=params,
        trainable_params=trainable_params,
        model_size_mb=size_all_mb
    )

def load_pretrained_models(model_names=None, device='cpu'):
    """
    Load multiple pre-trained models for comparison
    
    Args:
        model_names: List of model names to load, or None to load all available models
        device: Device to load models to
    
    Returns:
        Dictionary mapping model names to (model, transforms) tuples
    """
    if model_names is None:
        model_names = get_available_classification_models()
    
    models = {}
    for name in model_names:
        print(f"Loading pre-trained model: {name}")
        try:
            model, transforms = create_model_adapter(name, pretrained=True)
            if model is not None:
                model = model.to(device)
                models[name] = (model, transforms)
                print(f"  - Successfully loaded {name}")
            else:
                print(f"  - Failed to load {name}")
        except Exception as e:
            print(f"  - Error loading {name}: {e}")
    
    return models 