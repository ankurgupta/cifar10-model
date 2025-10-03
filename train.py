import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from model import CIFAR10Model

# Define global constants
SEED = 2
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


# find the gpu device to be used for training
def get_device():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("GPU Available?", device)
    return device


# setting the seed for reproducibility
def set_seed(seed, device):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)


# Dataset definition
class AlbumentationDataset(CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


# Define transformations
train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.1),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=(0.4914, 0.4822, 0.4465),
            mask_fill_value=None,
            p=0.1,
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)


# DataLoader creation
def create_dataloaders(train_transform, test_transform, batch_size, num_workers=4):
    train_dataset = AlbumentationDataset(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = AlbumentationDataset(
        root="./data", train=False, download=True, transform=test_transform
    )

    dataloader_args = dict(
        shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    train_loader = DataLoader(train_dataset, **dataloader_args)
    test_loader = DataLoader(test_dataset, **dataloader_args)

    return train_loader, test_loader


# Training and Testing Functions


def train(model, device, train_loader, optimizer, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, target)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100.0 * correct / len(test_loader.dataset))

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n"
    )

    # Main function


def main():
    device = get_device()
    set_seed(SEED, device)
    train_loader, test_loader = create_dataloaders(
        train_transforms, test_transforms, batch_size=128
    )
    model = CIFAR10Model().to(device)
    # summary(model, input_size=(3, 32, 32))
    # summary(CIFAR10Model().to("cpu"), input_size=(3, 32, 32))

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    log_table = PrettyTable()
    log_table.field_names = [
        "Epoch",
        "Training Accuracy",
        "Test Accuracy",
        "Diff",
        "Training Loss",
        "Test Loss",
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.8)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch + 1)
        train(model, device, train_loader, optimizer, train_losses, train_acc)
        scheduler.step()
        test(model, device, test_loader, test_losses, test_acc)
        log_table.add_row(
            [
                epoch + 1,
                f"{train_acc[-1]:.2f}%",
                f"{test_acc[-1]:.2f}%",
                f"{float(train_acc[-1]) - float(test_acc[-1]):.2f}",
                f"{train_losses[-1]:.4f}",
                f"{test_losses[-1]:.4f}",
            ]
        )
    print(log_table)


if __name__ == "__main__":
    main()
