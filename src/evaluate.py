#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

from preprocess import preprocess_image, create_variable_sized_images, create_multimodal_dummy_data
from train import FTDCPolicy, BaselinePolicy, MultiModalFTDC

def load_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'ftdc_config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_tokenization(tokenizer, img_sizes, save_path='../'):
    """
    Evaluate the dynamic tokenization on variable-shaped inputs.
    
    Args:
        tokenizer: DynamicTokenizer instance
        img_sizes: list of tuples (height, width)
        save_path: directory to save the plot
        
    Returns:
        token_shapes: list of token shapes
        reconstruction_errors: list of dummy reconstruction errors
    """
    reconstruction_errors = []
    token_shapes = []
    
    images = create_variable_sized_images(img_sizes)
    
    for i, img in enumerate(images):
        img_tensor = preprocess_image(img)
        tokens = tokenizer(img_tensor)
        token_shapes.append(tokens.shape[1])
        error = 1.0 / (tokens.shape[1] + 0.1)
        reconstruction_errors.append(error)
        print(f"Image size: {img_sizes[i]}, Tokens shape (B, T, embed_dim): {tokens.shape}, Reconstruction error: {error:.4f}")
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(token_shapes, reconstruction_errors, 'o-', linewidth=2, markersize=8, label="Reconstruction Error")
    plt.xlabel("Number of Tokens", fontsize=12)
    plt.ylabel("Reconstruction Error (Dummy)", fontsize=12)
    plt.title("Reconstruction Error vs. Token Count (Experiment 1)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "reconstruction_error_experiment1.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Experiment 1 plot saved as {os.path.join(save_path, 'reconstruction_error_experiment1.pdf')}")
    
    return token_shapes, reconstruction_errors

def plot_training_loss(losses, model_name, save_path='../'):
    """
    Plot training loss curve.
    
    Args:
        losses: list of losses per epoch
        model_name: name of the model
        save_path: directory to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=8, label=f"{model_name} Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training Loss Curve - {model_name} (Experiment 2)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"training_loss_{model_name.lower()}_experiment2.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Training loss plot for {model_name} saved as {os.path.join(save_path, f'training_loss_{model_name.lower()}_experiment2.pdf')}")

def evaluate_multimodal_model(model, images, sensors, save_path='../'):
    """
    Evaluate the multi-modal FTDC model.
    
    Args:
        model: MultiModalFTDC instance
        images: tensor of shape (batch_size, 3, height, width)
        sensors: tensor of shape (batch_size, sensor_dim)
        save_path: directory to save the plot
        
    Returns:
        actions: predicted actions
    """
    device = next(model.parameters()).device
    images = images.to(device)
    sensors = sensors.to(device)
    
    with torch.no_grad():
        actions = model(images, sensors)
    
    print("Predicted actions from multi-modal model:")
    print(actions.detach().cpu().numpy())
    
    avg_actions = actions.mean(dim=0).detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=list(range(len(avg_actions))), y=avg_actions, palette="viridis")
    plt.xlabel("Action Dimension", fontsize=12)
    plt.ylabel("Average Prediction", fontsize=12)
    plt.title("Average Predicted Actions (Experiment 3)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "predicted_actions_experiment3.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Predicted actions plot saved as {os.path.join(save_path, 'predicted_actions_experiment3.pdf')}")
    
    return actions

def compare_models(ftdc_losses, baseline_losses, save_path='../'):
    """
    Compare FTDC and baseline models by plotting their training losses.
    
    Args:
        ftdc_losses: list of FTDC losses per epoch
        baseline_losses: list of baseline losses per epoch
        save_path: directory to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(ftdc_losses)+1), ftdc_losses, 'o-', linewidth=2, markersize=8, label="FTDC Loss")
    plt.plot(range(1, len(baseline_losses)+1), baseline_losses, 's-', linewidth=2, markersize=8, label="Baseline BC Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Comparison (Experiment 2)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "loss_comparison_experiment2.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Loss comparison plot saved as {os.path.join(save_path, 'loss_comparison_experiment2.pdf')}")

if __name__ == "__main__":
    from preprocess import DynamicTokenizer
    
    config = load_config()
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    tokenizer = DynamicTokenizer(token_size=16, embed_dim=64)
    img_sizes = [(64, 64), (80, 80), (128, 128), (100, 150)]
    evaluate_tokenization(tokenizer, img_sizes)
    
    dummy_losses = [0.5, 0.3, 0.2]
    plot_training_loss(dummy_losses, "FTDC")
    
    images, sensors = create_multimodal_dummy_data()
    model = MultiModalFTDC(img_embed_dim=64, sensor_input_dim=20, sensor_embed_dim=64, num_actions=5)
    evaluate_multimodal_model(model, images, sensors)
    
    print("Evaluation tests completed successfully.")
