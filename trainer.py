import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNNTrainer:
    """CNN training class with comprehensive monitoring and callbacks"""

    def __init__(self, model, device, train_loader, val_loader, 
                 learning_rate=0.001, weight_decay=1e-4):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                         factor=0.5, patience=5)

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs, early_stopping_patience=10, save_path='best_model.pth'):
        """Train the model with early stopping and checkpointing"""

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
                print(f'✓ New best model saved (Val Acc: {val_acc:.2f}%)')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {patience_counter} epochs without improvement')
                break

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def plot_training_history(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss curves
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Loss vs Accuracy
        ax4.scatter(self.history['val_loss'], self.history['val_acc'], 
                   c=epochs, cmap='viridis', alpha=0.7)
        ax4.set_title('Validation Loss vs Accuracy')
        ax4.set_xlabel('Validation Loss')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.grid(True)

        plt.tight_layout()
        return fig
