#!/usr/bin/env python

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import os
import yaml
import sys

from preprocess import (
    DynamicTokenizer, 
    preprocess_image, 
    create_variable_sized_images, 
    create_dummy_dataloader,
    create_multimodal_dummy_data
)
from train import (
    FTDCPolicy, 
    BaselinePolicy, 
    MultiModalFTDC, 
    train_ftdc_model, 
    train_baseline_model
)
from evaluate import (
    evaluate_tokenization, 
    plot_training_loss, 
    evaluate_multimodal_model, 
    compare_models
)

def load_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'ftdc_config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_status(status, config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'ftdc_config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['status_enum'] = status
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Status updated to: {status}")

def experiment1(save_path=None):
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, 'results')
        os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*80)
    print("Running Experiment 1: Dynamic Tokenization on Variable-Shaped Inputs")
    print("="*80)
    
    config = load_config()
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_sizes = [(64, 64), (80, 80), (128, 128), (100, 150)]
    
    token_size = config['experiment']['token_size']
    embed_dim = config['experiment']['embed_dim']
    tokenizer = DynamicTokenizer(token_size=token_size, embed_dim=embed_dim).to(device)
    
    token_shapes, reconstruction_errors = evaluate_tokenization(tokenizer, img_sizes, save_path)
    
    print("\nExperiment 1 completed successfully.")
    return token_shapes, reconstruction_errors

def experiment2(save_path=None):
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, 'results')
        os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*80)
    print("Running Experiment 2: Diffusion-Driven Policy Learning vs. Conventional Behavioral Cloning")
    print("="*80)
    
    config = load_config()
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_actions = config['experiment']['num_actions']
    embed_dim = config['experiment']['embed_dim']
    num_epochs = config['experiment']['num_epochs']
    
    ftdc_model = FTDCPolicy(embed_dim=embed_dim, num_actions=num_actions).to(device)
    
    print("\nTraining FTDC (with diffusion) model:")
    ftdc_losses = train_ftdc_model(ftdc_model, create_dummy_dataloader, num_epochs=num_epochs, model_name="FTDC", device=device)
    
    plot_training_loss(ftdc_losses, "FTDC", save_path)
    
    baseline_model = BaselinePolicy(embed_dim=embed_dim, num_actions=num_actions).to(device)
    
    print("\nTraining Baseline BC model:")
    baseline_losses = train_baseline_model(baseline_model, create_dummy_dataloader, num_epochs=num_epochs, device=device)
    
    plot_training_loss(baseline_losses, "Baseline", save_path)
    
    compare_models(ftdc_losses, baseline_losses, save_path)
    
    print("\nExperiment 2 completed successfully.")
    return ftdc_losses, baseline_losses

def experiment3(save_path=None):
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, 'results')
        os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*80)
    print("Running Experiment 3: Flexible Transformer Backbone for Multi-Dimensional Data")
    print("="*80)
    
    config = load_config()
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = config['experiment']['batch_size']
    num_actions = 5  # Using 5 for this experiment to match the example code
    img_embed_dim = config['experiment']['embed_dim']
    sensor_input_dim = 20
    sensor_embed_dim = config['experiment']['embed_dim']
    
    images, sensors = create_multimodal_dummy_data(batch_size=batch_size)
    
    multimodal_model = MultiModalFTDC(
        img_embed_dim=img_embed_dim, 
        sensor_input_dim=sensor_input_dim, 
        sensor_embed_dim=sensor_embed_dim, 
        num_actions=num_actions
    ).to(device)
    
    actions = evaluate_multimodal_model(multimodal_model, images, sensors, save_path)
    
    print("\nExperiment 3 completed successfully.")
    return actions

def run_tests(save_path=None):
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, 'results')
        os.makedirs(save_path, exist_ok=True)
    
    start_time = time.time()
    print("\nStarting quick tests for all experiments...\n")
    
    experiment1(save_path)
    
    experiment2(save_path)
    
    experiment3(save_path)
    
    elapsed = time.time() - start_time
    print(f"\nAll experiments executed in {elapsed:.2f} seconds. Test finished successfully.")

if __name__ == '__main__':
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(save_path, exist_ok=True)
    
    update_status("running")
    
    try:
        print("\n" + "="*80)
        print("FTDC (Flexible Token Diffusion Cloning) Experiments")
        print("="*80)
        
        experiment1(save_path)
        
        experiment2(save_path)
        
        experiment3(save_path)
        
        print("\n" + "="*80)
        print("All experiments completed successfully!")
        print("="*80)
        
        update_status("stopped")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        update_status("stopped")
        sys.exit(1)
