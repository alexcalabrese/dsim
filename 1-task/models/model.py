"""
CNN-based emotion classification model with modular architecture
"""

import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """CNN architecture for audio emotion classification
    
    Architecture:
        - Input: (1, n_mfcc, time_steps)
        - 4 convolutional blocks with batch norm
        - Global average pooling
        - 2 fully connected layers with dropout
    """
    
    def __init__(self, input_shape, num_classes, conv_params=None):
        """
        Args:
            input_shape (tuple): (channels, height, width)
            num_classes (int): Number of output classes
            conv_params (list): List of convolution parameters
        """
        super().__init__()
        
        # Default convolution parameters
        self.conv_params = conv_params or [
            (32, 3, 1),  # (out_channels, kernel_size, stride)
            (64, 3, 1),
            (128, 3, 1),
            (256, 3, 1)
        ]
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_shape[0]
        
        for params in self.conv_params:
            out_channels, kernel_size, stride = params
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size//2
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = out_channels
            
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self._forward_features(dummy_input)
            self.feature_size = dummy_output.view(-1).shape[0]
            
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def _forward_features(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x) 