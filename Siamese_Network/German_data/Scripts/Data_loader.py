import os
import random
from PIL import Image
from torch.utils.data import Dataset

class Triplet_Loader(Dataset):
    def __init__(self, queries, reference_views, transforms=None, max_images=-1):
        self.queries = queries
        self.reference_views = reference_views
        self.transforms = transforms

        # Filter out non-png and files starting with '._'
        self.image_names = sorted([
            f for f in os.listdir(self.queries)
            if f.endswith('.png') and not f.startswith('._')
        ])

        # If max_images is specified, limit the number of images
        if max_images is not None and max_images > 0:
            self.image_names = self.image_names[:max_images]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        anchor_image_path = os.path.join(self.queries, self.image_names[idx])
        positive_image_path = os.path.join(self.reference_views, self.image_names[idx])

        # Randomly select a negative image
        negative_idx = random.choice(range(len(self.image_names)))
        while negative_idx == idx:
            negative_idx = random.choice(range(len(self.image_names)))
        negative_image_path = os.path.join(self.reference_views, self.image_names[negative_idx])

        # Load images
        anchor_image = Image.open(anchor_image_path).convert('RGB')
        positive_image = Image.open(positive_image_path).convert('RGB')
        negative_image = Image.open(negative_image_path).convert('RGB')

        # Apply transformations
        if self.transforms:
            anchor_image = self.transforms(anchor_image)
            positive_image = self.transforms(positive_image)
            negative_image = self.transforms(negative_image)

        return anchor_image, positive_image, negative_image
