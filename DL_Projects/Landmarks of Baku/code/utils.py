import torch
import torch.nn.functional as F
import time
import csv

def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=10, save_path=None):
    """
    Trains the model using the provided data loaders and optimizer.

    Parameters:
    - model (nn.Module): Model to train.
    - train_loader (DataLoader): Training dataset.
    - test_loader (DataLoader): Test dataset.
    - optimizer (torch.optim): Optimizer instance.
    - criterion (nn.Module): Loss function.
    - device (torch.device): CPU or CUDA.
    - epochs (int): Number of training epochs.
    - save_path (str): File path to save model state.

    Returns:
    - train_accuracies (List[float]): Training accuracy per epoch.
    - test_accuracies (List[float]): Test accuracy per epoch.
    """
    model.to(device)
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} — train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)

    return train_accuracies, test_accuracies

def evaluate(model, data_loader, criterion, device):
    """
    Evaluates model performance on a given dataset.

    Parameters:
    - model (nn.Module): Trained model.
    - data_loader (DataLoader): Validation or test data loader.
    - criterion (nn.Module): Loss function.
    - device (torch.device): CPU or GPU.

    Returns:
    - avg_loss (float): Average loss over dataset.
    - accuracy (float): Accuracy over dataset.
    """
    model.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = loss / len(data_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy