import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_fashion_loaders(batch_size=64):
    # here we convert the image to tensor and normalize it  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # here we download the dataset automatically via PyTorch 
    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # splitting the data (70% train, 15% val, 15% test)
    num_total = len(full_train_dataset)
    num_train = int(0.7 * num_total)
    num_val = int(0.15 * num_total)
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
    return train_loader, val_loader, test_loader, classes