import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    """VGG convolutional block with batch normalization and dropout"""
    
    def __init__(self, in_channels, out_channels, num_convs, dropout=0.3):
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGGNet(nn.Module):
    """VGG-inspired CNN for CIFAR-10 classification"""
    
    def __init__(self, num_classes=10, dropout=0.5):
        super(VGGNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            VGGBlock(3, 64, 2, dropout=0.25),
            
            # Block 2: 16x16 -> 8x8  
            VGGBlock(64, 128, 2, dropout=0.25),
            
            # Block 3: 8x8 -> 4x4
            VGGBlock(128, 256, 3, dropout=0.25),
            
            # Block 4: 4x4 -> 2x2
            VGGBlock(256, 512, 3, dropout=0.25),
            
            # Block 5: 2x2 -> 1x1
            VGGBlock(512, 512, 3, dropout=0.25)
        )
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_vgg_model(num_classes=10, dropout=0.5):
    """Factory function to create VGG model"""
    return VGGNet(num_classes=num_classes, dropout=dropout)

def print_model_summary(model, input_size=(3, 32, 32)):
    """Print model architecture summary"""
    print("Model Architecture:")
    print("-" * 60)
    print(model)
    print("-" * 60)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 60)
