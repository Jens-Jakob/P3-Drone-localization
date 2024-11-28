import csv
import torch
from model import SiameseNetwork
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import json  # Added to handle JSON operations

# ---------------------- Custom Dataset Class ---------------------- #
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_names = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) and not f.startswith('._')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_name
        except Exception as e:
            print(f"Warning: Error loading image {img_name}: {e}")
            return None, img_name

# ------------------ Helper Functions ------------------ #
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if batch:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None  # Return None if batch is empty

# -------------------- Device and Model Setup -------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
checkpoint_path = '/ceph/project/QSS/støvring_test/data/checks/triplet_model_epoch_30.pth'

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------------- Image Transformations ---------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Apply the same normalization used during training
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- Dataset and DataLoader Setup -------------------- #
query_folder = 'Støvring_UAV_test'
reference_folder = 'Støvring_satellite'

batch_size = 32
num_workers = 4

query_dataset = ImageDataset(query_folder, transform=transform)
reference_dataset = ImageDataset(reference_folder, transform=transform)

print(f"Number of images in query dataset: {len(query_dataset)}")
print(f"Number of images in reference dataset: {len(reference_dataset)}")

query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
reference_loader = DataLoader(reference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# --------------------- Embeddings Computation --------------------- #
def compute_embeddings(dataloader):
    embeddings = []
    image_names = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue  # Skip empty batches
            batch_imgs, batch_names = batch
            print(f"Processing batch {i+1}, batch size: {len(batch_imgs)}")
            batch_imgs = batch_imgs.to(device)
            output = model.forward_once(batch_imgs)
            embeddings.append(output)
            image_names.extend(batch_names)
    if not embeddings:
        print("No embeddings were computed. Please check your dataset and data loader.")
        return None, None
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, image_names

reference_embeddings, reference_image_names = compute_embeddings(reference_loader)
if reference_embeddings is None:
    print("Error computing reference embeddings.")
    exit()

query_embeddings, query_image_names = compute_embeddings(query_loader)
if query_embeddings is None:
    print("Error computing query embeddings.")
    exit()

# Normalize embeddings
reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

# ------------------- Create Mapping from CSV ------------------- #
def create_mapping(csv_file):
    mapping = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uav_image_name = row['UAVImageName']
            sat_image_name = row['SatelliteImageName']
            label = int(row['Label'])
            if label == 1:
                if uav_image_name not in mapping:
                    mapping[uav_image_name] = []
                # Adjust satellite image name to match actual filenames
                sat_image_name = adjust_satellite_image_name(sat_image_name)
                mapping[uav_image_name].append(sat_image_name)
    return mapping

def adjust_satellite_image_name(image_name):
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

# Provide the path to your CSV file used during training
csv_file = '../data/triplet_data.csv'  # Update this path
mapping = create_mapping(csv_file)

# --------------------- Evaluation --------------------- #
correct_matches_top1 = 0
correct_matches_top5 = 0
correct_matches_top10 = 0
correct_matches_top20 = 0
correct_matches_top35 = 0
correct_matches_top50 = 0

total_queries = 0

# Create a lookup for reference image indices
reference_image_indices = {name: idx for idx, name in enumerate(reference_image_names)}

# Dictionary to store top 50 matches for each query image
top_matches_data = {}

for i, query_name in enumerate(query_image_names):
    if query_name not in mapping:
        continue  # No matching reference images for this query
    total_queries += 1
    # Get the indices of matching reference images
    matching_reference_names = mapping[query_name]
    matching_reference_indices = [reference_image_indices.get(name) for name in matching_reference_names if name in reference_image_indices]
    if not matching_reference_indices:
        continue  # No matching reference images found in the reference set
    query_embedding = query_embeddings[i]
    # Compute similarities
    query_similarity = torch.mv(reference_embeddings, query_embedding)
    sorted_indices = torch.argsort(query_similarity, descending=True)
    # Check the ranks of matching reference images
    ranks = [sorted_indices.tolist().index(idx) + 1 for idx in matching_reference_indices if idx in sorted_indices]
    if not ranks:
        continue  # Matching images not found in the sorted list
    first_relevant_rank = min(ranks)
    # Top-1
    if first_relevant_rank == 1:
        correct_matches_top1 += 1
    # Top-5
    if first_relevant_rank <= 5:
        correct_matches_top5 += 1
    # Top-10
    if first_relevant_rank <= 10:
        correct_matches_top10 +=1
    # Top-20
    if first_relevant_rank <= 20:
        correct_matches_top20 +=1
    # Top-35
    if first_relevant_rank <= 35:
        correct_matches_top35 +=1
    # Top-50
    if first_relevant_rank <= 50:
        correct_matches_top50 +=1

    # Collect top 50 matches for this query image
    top_50_matches = []
    top_50_indices = sorted_indices[:50].cpu().numpy()
    for idx in top_50_indices:
        ref_name = reference_image_names[idx]
        similarity = query_similarity[idx].item()
        top_50_matches.append({
            'reference_image_name': ref_name,
            'similarity': similarity
        })

    # Store the top 50 matches in the dictionary with the query image name as the key
    top_matches_data[query_name] = top_50_matches

if total_queries == 0:
    print("No valid queries found. Please check your query dataset.")
else:
    top1_accuracy = correct_matches_top1 / total_queries * 100
    top5_accuracy = correct_matches_top5 / total_queries * 100
    top10_accuracy = correct_matches_top10 / total_queries * 100
    top20_accuracy = correct_matches_top20 / total_queries * 100
    top35_accuracy = correct_matches_top35 / total_queries * 100
    top50_accuracy = correct_matches_top50 / total_queries * 100

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Top-10 Accuracy: {top10_accuracy:.2f}%")
    print(f"Top-20 Accuracy: {top20_accuracy:.2f}%")
    print(f"Top-35 Accuracy: {top35_accuracy:.2f}%")
    print(f"Top-50 Accuracy: {top50_accuracy:.2f}%")

# --------------------- Save Top Matches --------------------- #
# Save the top matches data to a JSON file
with open('top_50_matches.json', 'w') as f:
    json.dump(top_matches_data, f, indent=4)

print("\nTop 50 matches for all query images saved to 'top_50_matches.json'.")
