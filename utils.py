import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

#### Methods to plot training and dataset metrics


def plot_dataset_distribution():
    counts = [11994, 105442, 13033]
    labels = ['Test', 'Train', 'Validation']

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    plt.title('Dataset Split Distribution', fontsize = 20, pad = 20)
    plt.axis('equal')  
    plt.show()

def label_frequency_histogram(bin_size):
    label_dict_path = './labels/train_dict.npy'
    labels_dict = np.load(label_dict_path, allow_pickle=True).item()
    all_labels = list(labels_dict.values())
    freq = Counter(all_labels)
    freqs = np.array(list(freq.values()))
    max_freq = freqs.max()
    min_freq = freqs.min()
    bin_edges = np.arange((min_freq // bin_size) * bin_size, max_freq + bin_size, bin_size)

    hist, edges = np.histogram(freqs, bins=bin_edges)
    bins = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    plt.figure(figsize=(10, 5))
    plt.bar(bins, hist, width=0.8, edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Number of samples per class")
    plt.ylabel("Number of Classes")
    plt.title("Class Frequency Distribution")
    plt.tight_layout()
    plt.show()
    return bins, hist

import torch
import matplotlib.pyplot as plt

def plot_training_metrics():

    checkpoint_path="./models/xtent_last_ssl_classification_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    epochs = range(1, len(checkpoint['train_loss']) + 1)

    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_accuracy']
    train_precision = checkpoint['train_precision']
    val_precision = checkpoint['val_precision']
    train_recall = checkpoint['train_recall']
    val_recall = checkpoint['val_recall']
    train_f1 = checkpoint['train_f1']
    val_f1 = checkpoint['val_f1']
    lrs = checkpoint.get('learning_rates', None)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth = 2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss For SSL (Trained on )")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", linewidth = 2)
    plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy For Supervised Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_precision, label="Train Precision", linewidth = 2)
    plt.plot(epochs, val_precision, label="Validation Precision", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Training vs Validation Precision For Supervised Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_recall, label="Train Recall", linewidth = 2)
    plt.plot(epochs, val_recall, label="Validation Recall", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Training vs Validation Recall For Supervised Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1, label="Train F1", linewidth = 2)
    plt.plot(epochs, val_f1, label="Validation F1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training vs Validation F1 Score For Supervised Learning")
    plt.legend()
    plt.grid(True)
    plt.show()

    if lrs is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, lrs, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule For Supervised Learning")
        plt.grid(True)
        plt.show()

    print("=== Final Test Metrics ===")
    print(f"Test Loss: {checkpoint['test_loss']:.4f}")
    print(f"Test Accuracy: {checkpoint['test_acc']:.4f}")
    print(f"Test Precision: {checkpoint['test_precision']:.4f}")
    print(f"Test Recall: {checkpoint['test_recall']:.4f}")
    print(f"Test F1: {checkpoint['test_f1']:.4f}")


def plot_pretext_loss():

    checkpoint_path="./models/xtent_last_ssl_pretext_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    epochs = range(1, len(checkpoint['train_loss']) + 1)

    train_loss = checkpoint['train_loss']
    

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth = 2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss For Pretext Task")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix():

    checkpoint = torch.load('.\\models\\take2_last_sup_classification_model.pth', weights_only=False,  map_location='cpu')
    cm = checkpoint.get('confusion_matrix_test', None)

    if cm is None:
        raise ValueError("Confusion matrix not found in checkpoint!")

    cm = np.array(cm)


    plt.figure(figsize=(10, 8))
    tick_labels = np.arange(1, 251 + 1)
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels)

    plt.title("Confusion Matrix on the Test Set for Supervised Learning")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()


def plot_training_metrics_comparison():
    # Load the two checkpoints
    checkpoint1 = torch.load('./models/xtent_last_ssl_classification_model.pth', map_location="cpu", weights_only=False)
    model1_label = 'Unfrozen Convolutional Layers'
    checkpoint2 = torch.load('./models/xtent_fc_last_ssl_classification_model.pth', map_location="cpu", weights_only=False)
    model2_label = 'Training only FC layers'
    # Metrics to compare
    metrics = [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc", "val_accuracy", "Accuracy"),
        ("train_precision", "val_precision", "Precision"),
        ("train_recall", "val_recall", "Recall"),
        ("train_f1", "val_f1", "F1 Score")
    ]
    
    epochs1 = range(1, len(checkpoint1['train_loss']) + 1)
    epochs2 = range(1, len(checkpoint2['train_loss']) + 1)
    
    # Plot each metric for both models
    for train_key, val_key, label in metrics:
        plt.figure(figsize=(10, 5))
        
        # Model 1
        plt.plot(epochs1, checkpoint1[train_key], label=f"{model1_label} Train {label}", linewidth=2, color = 'g')
        plt.plot(epochs1, checkpoint1[val_key], label=f"{model1_label} Val {label}", linewidth=2, linestyle="--", color = 'g')
        
        # Model 2
        plt.plot(epochs2, checkpoint2[train_key], label=f"{model2_label} Train {label}", linewidth=2, color = 'r')
        plt.plot(epochs2, checkpoint2[val_key], label=f"{model2_label} Val {label}", linewidth=2, linestyle="--", color = 'r')
        
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"Training vs Validation {label} Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plot Learning Rate schedule (if available)
    if 'learning_rates' in checkpoint1 or 'learning_rates' in checkpoint2:
        plt.figure(figsize=(10, 5))
        if 'learning_rates' in checkpoint1:
            plt.plot(epochs1, checkpoint1['learning_rates'], label=f"{model1_label} Learning Rate", linewidth=2)
        if 'learning_rates' in checkpoint2:
            plt.plot(epochs2, checkpoint2['learning_rates'], label=f"{model2_label} Learning Rate", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()


def print_test_metrics(checkpoint, label):
        print(f"=== Final Test Metrics ({label}) ===")
        print(f"Test Loss: {checkpoint['test_loss']:.4f}")
        print(f"Test Accuracy: {checkpoint['test_acc']:.4f}")
        print(f"Test Precision: {checkpoint['test_precision']:.4f}")
        print(f"Test Recall: {checkpoint['test_recall']:.4f}")
        print(f"Test F1: {checkpoint['test_f1']:.4f}\n")
    

if __name__ == "__main__":
    checkpoint1 = torch.load('./models/xtent_last_ssl_classification_model.pth', map_location="cpu", weights_only=False)
    model1_label = 'Unfrozen Convolutional Layers'
    checkpoint2 = torch.load('./models/xtent_fc_last_ssl_classification_model.pth', map_location="cpu", weights_only=False)
    model2_label = 'Training only FC layers'
    print_test_metrics(checkpoint1, model1_label)
    print_test_metrics(checkpoint2, model2_label)