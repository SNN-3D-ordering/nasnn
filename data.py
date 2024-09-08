import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size, data_path):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )

    mnist_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, test_loader
