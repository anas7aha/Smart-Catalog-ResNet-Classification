import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 1. Downloading only the test set (train=False) which the model has never seen during training
test_dataset = datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

# The names of the ten approved classes
classes = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

# create a folder to save the test images
output_dir = 'test_samples'
os.makedirs(output_dir, exist_ok=True)

# extract one unique and clean image from each class and save it
saved_classes = set()
for image, label in test_dataset:
    class_name = classes[label]
    if class_name not in saved_classes:
        # save image in PNG format inside the folder
        save_image(image, f'{output_dir}/{class_name}.png')
        saved_classes.add(class_name)
    
    # Stopping once the 10 complete classes are collected
    if len(saved_classes) == 10:
        break

print(f"✅ Success! 10 unseen test images saved inside '{output_dir}/' folder.")