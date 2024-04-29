from torchvision import datasets
from torch.utils.data import DataLoader


def create_data_loaders(train_path, test_path, data_transform, bach_size, num_workers):
    train_data = datasets.ImageFolder(train_path, data_transform, target_transform=None)
    test_data = datasets.ImageFolder(test_path, data_transform)
    train_dataloader = DataLoader(
        train_data, batch_size=bach_size, num_workers=num_workers, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=bach_size, num_workers=num_workers, shuffle=False
    )

    return train_dataloader, test_dataloader
