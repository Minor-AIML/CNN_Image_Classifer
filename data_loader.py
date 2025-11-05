import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class CIFAR10DataLoader:
    """CIFAR-10 data loader with preprocessing and augmentation"""

    def __init__(self, batch_size=128, validation_split=0.1, num_workers=2):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers

        # CIFAR-10 class names
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')

        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def get_data_loaders(self):
        """Get train, validation, and test data loaders"""

        # Load training data
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, 
            transform=self.train_transform
        )

        # Split training data into train and validation
        train_size = int((1 - self.validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply validation transform to validation set
        val_dataset.dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, 
            transform=self.val_test_transform
        )

        # Load test data
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, 
            transform=self.val_test_transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, 
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers, 
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def visualize_samples(self, data_loader, num_samples=16):
        """Visualize sample images from the dataset"""

        # Get a batch of data
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        images = images * std + mean
        images = torch.clamp(images, 0, 1)

        # Create subplot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('CIFAR-10 Sample Images', fontsize=16)

        for i in range(num_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[row, col].set_title(f'{self.classes[labels[i]]}')
            axes[row, col].axis('off')

        plt.tight_layout()
        return fig

    def get_class_distribution(self, data_loader):
        """Get class distribution in the dataset"""

        class_counts = torch.zeros(len(self.classes))

        for _, labels in data_loader:
            for label in labels:
                class_counts[label] += 1

        return {self.classes[i]: int(class_counts[i]) for i in range(len(self.classes))}

def calculate_dataset_statistics():
    """Calculate mean and std for CIFAR-10 dataset"""

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std
