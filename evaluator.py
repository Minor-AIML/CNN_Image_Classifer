import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm

class ModelEvaluator:
    """Comprehensive model evaluation class"""

    def __init__(self, model, device, test_loader, classes):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.classes = classes
        self.predictions = None
        self.true_labels = None
        self.predicted_labels = None

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("Evaluating model...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        self.predictions = np.array(all_outputs)
        self.true_labels = np.array(all_labels)
        self.predicted_labels = np.array(all_predictions)

        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""

        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predicted_labels, average='weighted'
        )

        # Per-class metrics
        class_report = classification_report(
            self.true_labels, self.predicted_labels, 
            target_names=self.classes, output_dict=True
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_report': class_report
        }

        return metrics

    def plot_confusion_matrix(self, normalize=True):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predicted_labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        return plt.gcf()

    def plot_class_performance(self):
        """Plot per-class performance metrics"""
        if not hasattr(self, 'true_labels'):
            raise ValueError("Model must be evaluated first")

        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predicted_labels, average=None
        )

        # Create DataFrame
        df = pd.DataFrame({
            'Class': self.classes,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Precision
        axes[0, 0].bar(df['Class'], df['Precision'], color='skyblue')
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Recall
        axes[0, 1].bar(df['Class'], df['Recall'], color='lightgreen')
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # F1-Score
        axes[1, 0].bar(df['Class'], df['F1-Score'], color='salmon')
        axes[1, 0].set_title('F1-Score by Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Support
        axes[1, 1].bar(df['Class'], df['Support'], color='gold')
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig, df

    def analyze_misclassifications(self, num_samples=20):
        """Analyze misclassified samples"""
        if not hasattr(self, 'true_labels'):
            raise ValueError("Model must be evaluated first")

        # Find misclassified indices
        misclassified_idx = np.where(self.true_labels != self.predicted_labels)[0]

        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return None

        # Get prediction probabilities for misclassified samples
        misclassified_probs = self.predictions[misclassified_idx]
        misclassified_true = self.true_labels[misclassified_idx]
        misclassified_pred = self.predicted_labels[misclassified_idx]

        # Sort by confidence (highest confidence misclassifications first)
        confidence_scores = np.max(misclassified_probs, axis=1)
        sorted_idx = np.argsort(confidence_scores)[::-1]

        # Create analysis DataFrame
        analysis_data = []
        for i in range(min(num_samples, len(sorted_idx))):
            idx = sorted_idx[i]
            analysis_data.append({
                'Sample_Index': misclassified_idx[idx],
                'True_Class': self.classes[misclassified_true[idx]],
                'Predicted_Class': self.classes[misclassified_pred[idx]],
                'Confidence': confidence_scores[idx],
                'True_Class_Prob': misclassified_probs[idx][misclassified_true[idx]],
                'Pred_Class_Prob': misclassified_probs[idx][misclassified_pred[idx]]
            })

        return pd.DataFrame(analysis_data)

    def plot_prediction_confidence(self):
        """Plot prediction confidence distribution"""
        if not hasattr(self, 'predictions'):
            raise ValueError("Model must be evaluated first")

        # Get confidence scores (max probability)
        confidence_scores = np.max(self.predictions, axis=1)
        correct_predictions = (self.true_labels == self.predicted_labels)

        # Separate correct and incorrect predictions
        correct_confidence = confidence_scores[correct_predictions]
        incorrect_confidence = confidence_scores[~correct_predictions]

        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidence, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_confidence, incorrect_confidence], 
                   labels=['Correct', 'Incorrect'])
        plt.ylabel('Prediction Confidence')
        plt.title('Confidence by Prediction Correctness')

        plt.tight_layout()
        return plt.gcf()

    def generate_evaluation_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        if not hasattr(self, 'true_labels'):
            metrics = self.evaluate_model()
        else:
            metrics = self._calculate_metrics()

        report = f"""
=== MODEL EVALUATION REPORT ===

Overall Performance:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}

Per-Class Performance:
"""

        for class_name in self.classes:
            if class_name in metrics['class_report']:
                class_metrics = metrics['class_report'][class_name]
                report += f"""
{class_name}:
  - Precision: {class_metrics['precision']:.4f}
  - Recall: {class_metrics['recall']:.4f}
  - F1-Score: {class_metrics['f1-score']:.4f}
  - Support: {class_metrics['support']}"""

        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report
