from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import batch_size, image_size

def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.Flowers102(root="./data", split="train", transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
