"""
=============================================================================
Project 03: Handwritten Digit Recognition on MNIST Dataset in PyTorch
=============================================================================
Author     : Gideon Mwiti
Challenge  : 100 PyTorch Projects in 10 Days
Topic      : Computer Vision, Image DataLoaders, Feedforward Neural Networks (MLP)
Goal       : Classify 28x28 grayscale handwritten digits (0-9) using PyTorch
=============================================================================
"""

import os
import sys
import types

# Ensure safe import of torchvision under all environments
sys.modules['bz2'] = types.SimpleNamespace(open=open)
sys.modules['lzma'] = types.SimpleNamespace(open=open)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class MNISTFeedforwardNN(nn.Module):
    """
    Deep Feedforward Neural Network (Multi-Layer Perceptron) for MNIST:
    - Flatten: 28x28 -> 784
    - Layer 1: 784 -> 256 (BatchNorm, ReLU, Dropout)
    - Layer 2: 256 -> 128 (BatchNorm, ReLU, Dropout)
    - Layer 3: 128 -> 10 (Output Logits for Classes 0 to 9)
    """
    def __init__(self, input_dim: int = 784, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10, dropout_rate: float = 0.2):
        super(MNISTFeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_data_loaders(data_dir: str, batch_size: int = 64):
    """
    Builds optimized training and testing DataLoaders with standard
    MNIST pixel normalization (mean=0.1307, std=0.3081).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader, test_dataset

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Executes a single training epoch and returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates the model on test data, returning loss, accuracy, and confusion matrix."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_loss = running_loss / total
    test_acc = (correct / total) * 100.0

    # Build 10x10 confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    return test_loss, test_acc, cm, np.array(all_preds), np.array(all_targets)

def plot_comprehensive_results(
    history: dict,
    cm: np.ndarray,
    test_dataset,
    model: nn.Module,
    device: torch.device,
    output_path: str
):
    """
    Renders a 4-panel diagnostic visual artifact:
    1. Sample test predictions with confidence levels
    2. Training vs Validation Cross-Entropy Loss
    3. Training vs Validation Accuracy (%)
    4. 10x10 Confusion Matrix Heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Project 03: MNIST Handwritten Digit Classifier (PyTorch Feedforward NN)\n"
        f"Final Test Accuracy = {history['test_acc'][-1]:.2f}% | Final Test Loss = {history['test_loss'][-1]:.4f}",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # 1. Sample Predictions Grid (10 sample digits)
    ax1 = axes[0, 0]
    ax1.axis("off")
    ax1.set_title("1. Model Predictions on Unseen Test Samples", fontweight="bold", fontsize=11, pad=12)

    # Pick 10 representative test samples
    indices = [3, 2, 1, 18, 4, 15, 11, 0, 61, 9]
    sample_subfig = ax1.inset_axes([0.0, 0.0, 1.0, 1.0])
    sample_subfig.axis("off")

    for idx_pos, img_idx in enumerate(indices):
        raw_img, true_label = test_dataset[img_idx]
        with torch.no_grad():
            model.eval()
            output = model(raw_img.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            pred_label = int(np.argmax(probs))
            conf = probs[pred_label] * 100.0

        # Subplot in 2 rows x 5 columns
        row = idx_pos // 5
        col = idx_pos % 5
        sub_ax = sample_subfig.inset_axes([col * 0.20 + 0.01, (1 - row) * 0.48, 0.17, 0.44])
        # Un-normalize image for clean display
        display_img = raw_img.squeeze().numpy() * 0.3081 + 0.1307
        sub_ax.imshow(display_img, cmap="gray")
        sub_ax.axis("off")
        color = "#059669" if pred_label == true_label else "#DC2626"
        sub_ax.set_title(f"True: {true_label} | Pred: {pred_label}\n({conf:.1f}%)", fontsize=8, color=color, fontweight="bold")

    # 2. Cross-Entropy Loss Curves
    ax2 = axes[0, 1]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax2.plot(epochs_range, history["train_loss"], color="#2563EB", linewidth=2.2, label="Train Loss", marker="o")
    ax2.plot(epochs_range, history["test_loss"], color="#EF4444", linewidth=2.2, label="Test Loss", marker="s")
    ax2.set_title("2. Cross-Entropy Loss Convergence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Accuracy Progression Curves
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, history["train_acc"], color="#2563EB", linewidth=2.2, label="Train Acc (%)", marker="o")
    ax3.plot(epochs_range, history["test_acc"], color="#10B981", linewidth=2.2, label="Test Acc (%)", marker="s")
    ax3.set_title("3. Classification Accuracy Progression (%)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim([90, 100])
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. 10x10 Confusion Matrix Heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title("4. Multi-Class Confusion Matrix (Digits 0-9)", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    tick_marks = np.arange(10)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(range(10))
    ax4.set_yticklabels(range(10))
    ax4.set_xlabel("Predicted Label")
    ax4.set_ylabel("Ground Truth Digit")

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            ax4.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontsize=7 if val > 999 else 8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visual report saved successfully at: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_plot_path = os.path.join(script_dir, "mnist_classification_results.png")

    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print("      PROJECT 03: MNIST DIGIT CLASSIFIER IN PYTORCH (FNN/MLP)    ")
    print("=================================================================")
    print(f"Execution Device    : {DEVICE}")
    print(f"Dataset             : MNIST (60,000 Train, 10,000 Test Grayscale 28x28)")
    print(f"Hyperparameters     : Epochs = {EPOCHS}, Batch Size = {BATCH_SIZE}, LR = {LEARNING_RATE}")
    print("-----------------------------------------------------------------")

    # Step 1: Load Data
    train_loader, test_loader, test_dataset = get_data_loaders(data_dir=data_dir, batch_size=BATCH_SIZE)

    # Step 2: Initialize Architecture
    model = MNISTFeedforwardNN(input_dim=784, hidden1=256, hidden2=128, num_classes=10, dropout_rate=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Test Loss':^12} | {'Test Acc (%)':^14}")
    print("-----------------------------------------------------------------")

    # Step 3: Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, cm, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {test_loss:^12.4f} | {test_acc:^14.2f}")

    print("-----------------------------------------------------------------")
    print(f"Training Complete! Final Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    print("=================================================================\n")

    # Step 4: Render Visual Artifact
    plot_comprehensive_results(
        history=history,
        cm=cm,
        test_dataset=test_dataset,
        model=model,
        device=DEVICE,
        output_path=output_plot_path
    )

if __name__ == "__main__":
    main()

