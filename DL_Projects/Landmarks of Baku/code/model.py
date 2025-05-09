from torchvision import models
import torch.nn as nn

def get_model(num_classes, dropout=0.0):
    """
    Loads a pretrained ResNet-18 model and replaces the final FC layer
    with a new classifier for custom number of classes.

    Parameters:
    - num_classes (int): Number of output classes.
    - dropout (float): Dropout value before the final classification layer.

    Returns:
    - model (nn.Module): Modified ResNet-18 model ready for training.
    """

    # Load a ResNet-18 model pretrained on ImageNet
    model = models.resnet18(pretrained=True)

    # Freeze all layers to keep pretrained weights intact
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )

    return model