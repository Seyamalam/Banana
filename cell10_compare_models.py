import os
import torch
import argparse
import torch.nn as nn

from cell1_imports_and_constants import set_seed, NUM_CLASSES
from cell2_dataset import load_data
from cell3_model import BananaLeafCNN
from cell8_model_zoo import load_pretrained_models, get_available_classification_models
from cell9_model_comparison import compare_models_with_pretrained

def parse_args():
    parser = argparse.ArgumentParser(description='Banana Leaf Disease Classification - Model Comparison')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Path to custom models directory')
    parser.add_argument('--results_dir', type=str, default='models/comparison_results', help='Path to save comparison results')
    parser.add_argument('--pretrained_models', type=str, default=None, help='Comma-separated list of pretrained models to compare with, or "all" for all available models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_formats', type=str, default='png,svg', help='Formats to save visualizations (comma-separated)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse save formats
    save_formats = args.save_formats.split(',')
    
    # Parse pretrained models
    if args.pretrained_models is None or args.pretrained_models.lower() == 'none':
        pretrained_model_names = []
    elif args.pretrained_models.lower() == 'all':
        pretrained_model_names = get_available_classification_models()
    else:
        pretrained_model_names = [name.strip() for name in args.pretrained_models.split(',')]
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data only (no need for training data in comparison)
    _, test_loader = load_data(args.data_dir, batch_size=args.batch_size)
    
    # Load pretrained models if specified
    if pretrained_model_names:
        print(f"Loading {len(pretrained_model_names)} pretrained models: {', '.join(pretrained_model_names)}")
        pretrained_models = load_pretrained_models(pretrained_model_names, device)
    else:
        print("No pretrained models specified for comparison")
        pretrained_models = None
    
    # Run comparison
    print("\nComparing custom models with pretrained models...")
    comparer = compare_models_with_pretrained(
        custom_model_dir=args.models_dir,
        results_dir=args.results_dir,
        test_loader=test_loader,
        device=device,
        pretrained_models=pretrained_models,
        save_formats=save_formats
    )
    
    print("\nModel comparison completed. Results saved to:", args.results_dir)
    print("\nSummary of compared models:")
    for i, model_info in enumerate(comparer.models_info):
        print(f"  {i+1}. {model_info.model_name} ({model_info.model_type})")
        print(f"     - Parameters: {model_info.params:,} (trainable: {model_info.trainable_params:,})")
        print(f"     - Size: {model_info.model_size_mb:.2f} MB")
        if model_info.accuracy is not None:
            print(f"     - Accuracy: {model_info.accuracy:.2f}%")
        if model_info.inference_time is not None:
            print(f"     - Inference time: {model_info.inference_time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    main() 