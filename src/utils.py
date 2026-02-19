import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision import transforms

# Predefinitions
cifar10_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Basic transforms for the FFNN
ffnn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales pixels to [-1, 1]
])

# Augmented transforms for the CNN
cnn_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def display_image(img_array, max_size=2):
    """
    Displays an image at maximum size while preserving aspect ratio.
    Removes axes and margins for a clean look.

    :param img_array: NumPy array image (e.g. from OpenCV)
    :param max_size: Maximum size (in inches) for the longest side
    """
    h, w = img_array.shape[:2]
    scale = max_size / max(h, w)
    figsize = (w * scale, h * scale)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.imshow(img_array, cmap='gray' if img_array.ndim == 2 else None, vmin=0, vmax=255)

    ax.set_axis_off()
    fig.patch.set_facecolor('none')
    ax.set_frame_on(False)
    plt.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)

    plt.show()

# Utility Image Function
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Loss Plot Function
def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.show()

# Accuracy Plot Function
def plot_acc(train_accs, val_accs):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    ax.plot(train_accs, label='Train Acc')
    ax.plot(val_accs, label='Val Acc')
    ax.set_title('Accuracy Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()

    plt.show()

# Plot Results from a DataFrame

# Plotted Validation Curves Function
def plot_validation_curves(histories_dict):
    plt.figure(figsize=(12, 8))

    # Loop through the dictionary
    for identifier, history in histories_dict.items():
        # Handle cases where the key might be a model object or a string
        label_name = identifier if isinstance(identifier, str) else identifier.__class__.__name__

        # Get the validation accuracy list
        val_accs = history['val_acc']
        epochs = range(1, len(val_accs) + 1)

        # Plot
        plt.plot(epochs, val_accs, marker='o', linestyle='-', label=f'{label_name}')

        # Annotate the final point
        final_acc = val_accs[-1]
        plt.annotate(f'{final_acc:.1f}%',
                     xy=(epochs[-1], final_acc),
                     xytext=(5, 0), textcoords='offset points')

    plt.title('Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Per-Class Accuracy Function
def per_class_accuracy(model, loader, device, classes):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print("\n--- Per-Class Accuracy ---")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'{classname:5s}: {accuracy:.1f} %')
