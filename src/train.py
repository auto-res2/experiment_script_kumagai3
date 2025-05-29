#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import os

from preprocess import DynamicTokenizer, create_dummy_dataloader

def load_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'ftdc_config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class SimpleDiffusionModule(nn.Module):
    """
    A simple diffusion module that predicts noise added to tokens.
    """
    def __init__(self, embed_dim):
        super(SimpleDiffusionModule, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, tokens):
        x = F.relu(self.fc1(tokens))
        noise_pred = self.fc2(x)
        return noise_pred

class FTDCPolicy(nn.Module):
    """
    Flexible Token Diffusion Cloning (FTDC) policy model.
    Combines dynamic tokenization, transformer backbone, and diffusion module.
    """
    def __init__(self, embed_dim, num_actions):
        super(FTDCPolicy, self).__init__()
        self.tokenizer = DynamicTokenizer(token_size=16, embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.diffusion = SimpleDiffusionModule(embed_dim=embed_dim)
        self.policy_head = nn.Linear(embed_dim, num_actions)
        
    def forward(self, x):
        tokens = self.tokenizer(x)                  # shape: (B, T, embed_dim)
        tokens = self.transformer(tokens.transpose(0,1)).transpose(0,1)
        noise_pred = self.diffusion(tokens)
        aggregated = tokens.mean(dim=1)
        actions = self.policy_head(aggregated)
        return actions, noise_pred

class BaselinePolicy(nn.Module):
    """
    Baseline behavioral cloning policy without diffusion module.
    Used for comparison with FTDC.
    """
    def __init__(self, embed_dim, num_actions):
        super(BaselinePolicy, self).__init__()
        self.tokenizer = DynamicTokenizer(token_size=16, embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.policy_head = nn.Linear(embed_dim, num_actions)
        
    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = self.transformer(tokens.transpose(0,1)).transpose(0,1)
        aggregated = tokens.mean(dim=1)
        actions = self.policy_head(aggregated)
        return actions

class MultiModalFTDC(nn.Module):
    """
    Multi-modal FTDC model that can process both image and sensor data.
    """
    def __init__(self, img_embed_dim, sensor_input_dim, sensor_embed_dim, num_actions):
        super(MultiModalFTDC, self).__init__()
        self.image_tokenizer = DynamicTokenizer(token_size=16, embed_dim=img_embed_dim)
        self.sensor_embed = nn.Linear(sensor_input_dim, sensor_embed_dim)
        
        fusion_dim = img_embed_dim  # assuming sensor_embed_dim is made to equal img_embed_dim
        self.pos_encoding_image = nn.Parameter(torch.randn(1, 100, fusion_dim))  # support up to 100 image tokens
        self.pos_encoding_sensor = nn.Parameter(torch.randn(1, 10, fusion_dim))  # support up to 10 sensor tokens
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.policy_head = nn.Linear(fusion_dim, num_actions)
    
    def forward(self, image, sensor):
        img_tokens = self.image_tokenizer(image)  # (B, T_img, fusion_dim)
        B, T_img, _ = img_tokens.shape
        pos_img = self.pos_encoding_image[:, :T_img, :]
        img_tokens = img_tokens + pos_img
        
        sensor_feat = self.sensor_embed(sensor)  # (B, sensor_embed_dim)
        sensor_tokens = sensor_feat.unsqueeze(1)   # (B, 1, fusion_dim)
        pos_sensor = self.pos_encoding_sensor[:, :sensor_tokens.size(1), :]
        sensor_tokens = sensor_tokens + pos_sensor
        
        fused_tokens = torch.cat([img_tokens, sensor_tokens], dim=1)  # (B, T_total, fusion_dim)
        fused_tokens = fused_tokens.transpose(0, 1)  # (T_total, B, fusion_dim)
        fused_tokens = self.transformer(fused_tokens)
        fused_tokens = fused_tokens.transpose(0, 1)  # (B, T_total, fusion_dim)
        
        aggregated = fused_tokens.mean(dim=1)
        actions = self.policy_head(aggregated)
        return actions

def train_ftdc_model(model, dataloader, num_epochs=3, model_name="FTDC", device="cuda"):
    """
    Train the FTDC model with both BC and diffusion losses.
    
    Args:
        model: FTDC model
        dataloader: data generator yielding (images, actions)
        num_epochs: number of training epochs
        model_name: name of the model for logging
        device: device to train on (cuda or cpu)
        
    Returns:
        epoch_losses: list of average losses per epoch
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epoch_losses = []
    
    for epoch in range(num_epochs):
        batch_losses = []
        for images, actions_gt in dataloader():
            images = images.to(device)
            actions_gt = actions_gt.to(device)
            
            actions_pred, noise_pred = model(images)
            bc_loss = F.mse_loss(actions_pred, actions_gt)
            diffusion_loss = F.mse_loss(noise_pred, torch.zeros_like(noise_pred))
            total_loss = bc_loss + 0.1 * diffusion_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_losses.append(total_loss.item())
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        print(f"{model_name} Epoch {epoch+1}/{num_epochs}: Avg Loss = {epoch_loss:.6f}")
        
    return epoch_losses

def train_baseline_model(model, dataloader, num_epochs=3, device="cuda"):
    """
    Train the baseline BC model with only BC loss.
    
    Args:
        model: Baseline BC model
        dataloader: data generator yielding (images, actions)
        num_epochs: number of training epochs
        device: device to train on (cuda or cpu)
        
    Returns:
        epoch_losses: list of average losses per epoch
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epoch_losses = []
    
    for epoch in range(num_epochs):
        batch_losses = []
        for images, actions_gt in dataloader():
            images = images.to(device)
            actions_gt = actions_gt.to(device)
            
            actions_pred = model(images)
            bc_loss = F.mse_loss(actions_pred, actions_gt)
            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()
            batch_losses.append(bc_loss.item())
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        print(f"Baseline BC Epoch {epoch+1}/{num_epochs}: Avg Loss = {epoch_loss:.6f}")
        
    return epoch_losses

if __name__ == "__main__":
    config = load_config()
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    embed_dim = config['experiment']['embed_dim']
    num_actions = config['experiment']['num_actions']
    
    ftdc_model = FTDCPolicy(embed_dim=embed_dim, num_actions=num_actions)
    
    dataloader = create_dummy_dataloader
    
    train_ftdc_model(ftdc_model, dataloader, num_epochs=1, device=device)
    
    print("FTDC model training test completed successfully.")
