import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, image_size=224, batch_size=32, augmented=False):
    """
    Loads training and testing datasets using torchvision's ImageFolder.
    Applies basic or augmented transforms based on the flag.

    Parameters:
    - data_dir (str): Path to the dataset folder.
    - image_size (int): Target size for resizing images.
    - batch_size (int): Number of images per batch.
    - augmented (bool): Whether to apply data augmentation on training data.

    Returns:
    - train_loader (DataLoader): DataLoader for training set.
    - test_loader (DataLoader): DataLoader for test set.
    - class_names (List[str]): List of class labels.
    """

    # Normalization for pre-trained models (ImageNet mean and std)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Training transforms
    if augmented:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])

    # Validation/test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Assumes the folder structure: data_dir/train and data_dir/test
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_dataset.classes