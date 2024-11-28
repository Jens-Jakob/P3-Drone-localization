import os
import random
import csv
from PIL import Image
from torch.utils.data import Dataset

class TripletDatasetFromCSV(Dataset):
    def __init__(self, csv_file, uav_image_root, satellite_image_root, transforms=None):
        self.transforms = transforms
        self.uav_image_root = uav_image_root
        self.satellite_image_root = satellite_image_root

        # Read the CSV file and build the data structure
        self.data = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uav_image_name = row['UAVImageName']
                sat_image_name = row['SatelliteImageName']
                label = int(row['Label'])

                if label == 1:
                    if uav_image_name not in self.data:
                        self.data[uav_image_name] = []
                    self.data[uav_image_name].append(sat_image_name)
                else:
                    pass  # Since negatives are not in the CSV

        # Build a list of all satellite images
        self.all_satellite_images = os.listdir(satellite_image_root)

        # List of UAV images that have at least one positive
        self.uav_images = list(self.data.keys())

    def __len__(self):
        return len(self.uav_images)

    def __getitem__(self, idx):
        # Get the UAV image name (anchor)
        uav_image_name = self.uav_images[idx]
        uav_image_path = os.path.join(self.uav_image_root, uav_image_name)

        # Randomly select a positive satellite image
        positive_image_name_original = random.choice(self.data[uav_image_name])
        positive_image_name = self.adjust_satellite_image_name(positive_image_name_original)
        positive_image_path = os.path.join(self.satellite_image_root, positive_image_name)

        # Randomly select a negative satellite image (not in positives)
        positive_image_names_adjusted = [self.adjust_satellite_image_name(name) for name in self.data[uav_image_name]]
        negative_candidates = [img for img in self.all_satellite_images if img not in positive_image_names_adjusted]
        if not negative_candidates:
            raise ValueError(f"No negative satellite images available for UAV image {uav_image_name}")
        negative_image_name = random.choice(negative_candidates)
        negative_image_path = os.path.join(self.satellite_image_root, negative_image_name)

        # Load images
        anchor_image = self.load_image(uav_image_path, image_type='uav')
        positive_image = self.load_image(positive_image_path, image_type='satellite')
        negative_image = self.load_image(negative_image_path, image_type='satellite')

        # Apply transformations
        if self.transforms:
            anchor_image = self.transforms(anchor_image)
            positive_image = self.transforms(positive_image)
            negative_image = self.transforms(negative_image)

        return anchor_image, positive_image, negative_image

    def adjust_satellite_image_name(self, image_name):
        if '_patch_' not in image_name:
            # Split the filename and extension
            name_without_ext, ext = os.path.splitext(image_name)
            # Split by '_'
            parts = name_without_ext.split('_')
            if len(parts) >= 2:
                # Insert '_patch_' before the last part
                new_name = '_'.join(parts[:-1]) + '_patch_' + parts[-1] + ext
                return new_name
        return image_name

    def load_image(self, image_path, image_type):
        try:
            img = Image.open(image_path)
            # Handle TIFF images
            if image_type == 'satellite' and img.format == 'TIFF':
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            return img
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {e}")
