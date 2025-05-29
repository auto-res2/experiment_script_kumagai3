#!/usr/bin/env python

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

class DynamicTokenizer(nn.Module):
    """
    Dynamic tokenizer that can handle variable-shaped inputs.
    Converts images into tokens based on their intrinsic dimensions.
    """
    def __init__(self, token_size, embed_dim):
        super(DynamicTokenizer, self).__init__()
        self.token_size = token_size  # e.g., 16 (16x16 patch)
        self.embed_dim = embed_dim
        self.token_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                                     kernel_size=token_size, stride=token_size)
        
    def forward(self, x):
        B, C, H, W = x.size()
        num_tokens_h = H // self.token_size
        num_tokens_w = W // self.token_size
        new_H, new_W = num_tokens_h * self.token_size, num_tokens_w * self.token_size
        x_cropped = x[:, :, :new_H, :new_W]
        tokens = self.token_embed(x_cropped)  # shape: (B, embed_dim, T_h, T_w)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens

def preprocess_image(image):
    """
    Convert BGR image to RGB and then to tensor.
    
    Args:
        image: numpy array of shape (H, W, C) in BGR format
        
    Returns:
        tensor: PyTorch tensor of shape (1, C, H, W)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ToTensor()(image)
    return image.unsqueeze(0)  # add batch dimension

def create_variable_sized_images(img_sizes):
    """
    Create a list of random images with different sizes.
    
    Args:
        img_sizes: list of tuples (height, width)
        
    Returns:
        images: list of numpy arrays with different sizes
    """
    images = []
    for size in img_sizes:
        fake_img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        images.append(fake_img)
    return images

def create_dummy_dataloader(batch_size=4, num_batches=5):
    """
    Create a dummy dataloader for simulation.
    
    Args:
        batch_size: number of samples per batch
        num_batches: number of batches to generate
        
    Returns:
        generator: yields batches of (images, actions)
    """
    for _ in range(num_batches):
        images = torch.randn(batch_size, 3, 64, 64)
        actions_gt = torch.randn(batch_size, 3)  # assume 3-dimensional action space
        yield images, actions_gt

def create_multimodal_dummy_data(batch_size=4, img_size=(64, 64), sensor_dim=20):
    """
    Create dummy data for multi-modal experiment.
    
    Args:
        batch_size: number of samples
        img_size: tuple (height, width) for images
        sensor_dim: dimension of sensor data
        
    Returns:
        images: tensor of shape (batch_size, 3, height, width)
        sensors: tensor of shape (batch_size, sensor_dim)
    """
    images = torch.randn(batch_size, 3, img_size[0], img_size[1])
    sensors = torch.randn(batch_size, sensor_dim)
    return images, sensors

if __name__ == "__main__":
    img_sizes = [(64, 64), (80, 80), (128, 128), (100, 150)]
    images = create_variable_sized_images(img_sizes)
    
    for i, img in enumerate(images):
        img_tensor = preprocess_image(img)
        tokenizer = DynamicTokenizer(token_size=16, embed_dim=64)
        tokens = tokenizer(img_tensor)
        print(f"Image size: {img_sizes[i]}, Tokens shape: {tokens.shape}")
