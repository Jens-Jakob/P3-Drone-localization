import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import torch


def data_loader(image_list, max_images=100):
    """Get the first 'max_images' from the original list."""
    return image_list[:max_images]


def load_and_preprocess_image(img_path):
    """ Load and preprocess the image for ResNet model """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension


        return img
    except UnidentifiedImageError:
        print(f"Could not identify image: {img_path}")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

    return None  # Return None if loading fails


def data_loader2(image_list, max_images=100):
    """Filter the image list to include only images with names from 00001.png to max_images."""
    selected_image_names = [f"{str(i).zfill(5)}.png" for i in range(1, max_images + 1)]

    # Filter the image list to include only those with the chosen names
    filtered_images = [img_path for img_path in image_list if os.path.basename(img_path) in selected_image_names]

    return filtered_images
