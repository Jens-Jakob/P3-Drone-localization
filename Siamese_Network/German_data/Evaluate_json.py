import csv
import torch
from Model import SiameseNetwork
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
            if f.lower().endswith('.png') and not f.startswith('._')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_name
        except Exception as e:
            print(f"Warning: Error loading image {img_name}: {e}")
            return None, img_name

# ------------------ Helper Functions ------------------ #
def extract_id(image_name):
    match = re.search(r'(\d+)', image_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# -------------------- Device and Model Setup -------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
checkpoint_path = '/ceph/project/QSS/Tysk_data/final/checks/50_triplet_model_epoch_epoch_15.pth'

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
query_folder = 'drone_images'
reference_folder = 'reference_views2'

query_batch_size = 32  # Process multiple query images at once
reference_batch_size = 32
num_workers = 14

query_dataset = ImageDataset(query_folder, transform=transform)
reference_dataset = ImageDataset(reference_folder, transform=transform)

# Remove the limitation to process only the first query image
# query_dataset.image_names = query_dataset.image_names[0:1]

query_loader = DataLoader(query_dataset, batch_size=query_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
reference_loader = DataLoader(reference_dataset, batch_size=reference_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# --------------------- Embeddings Computation --------------------- #
def compute_embeddings(dataloader):
    embeddings = []
    image_names = []
    image_ids = []
    with torch.no_grad():
        for batch_imgs, batch_names in dataloader:
            batch_imgs = batch_imgs.to(device)
            output = model.forward_once(batch_imgs)
            embeddings.append(output)
            image_names.extend(batch_names)
            batch_ids = [extract_id(name) for name in batch_names]
            image_ids.extend(batch_ids)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, image_names, image_ids

reference_embeddings, reference_image_names, reference_image_ids = compute_embeddings(reference_loader)
query_embeddings, query_image_names, query_image_ids = compute_embeddings(query_loader)

# Normalize embeddings if necessary
# reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)
# query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

# --------------------- Similarity Computation --------------------- #
# Use Euclidean distance
distances = torch.cdist(query_embeddings, reference_embeddings, p=2)

# --------------------- Evaluation --------------------- #
correct_matches_top1 = 0
correct_matches_top5 = 0
correct_matches_top10 = 0
correct_matches_top20 = 0
correct_matches_top35 = 0
correct_matches_top50 = 0

total_queries = len(query_image_ids)

# Dictionary to store top 20 matches for each query image
top_matches_data = {}

for i, query_id in enumerate(query_image_ids):
    if query_id == -1:
        continue
    query_distance = distances[i]
    sorted_indices = torch.argsort(query_distance)
    sorted_ids = [reference_image_ids[idx] for idx in sorted_indices.cpu().numpy()]
    # Top-1
    if sorted_ids[0] == query_id:
        correct_matches_top1 += 1
    # Top-5
    if query_id in sorted_ids[:5]:
        correct_matches_top5 += 1
    if query_id in sorted_ids[:10]:
        correct_matches_top10 +=1
    if query_id in sorted_ids[:20]:
        correct_matches_top20 +=1
    if query_id in sorted_ids[:35]:
        correct_matches_top35 +=1
    if query_id in sorted_ids[:50]:
        correct_matches_top50 +=1

    # Collect top 20 matches for this query image
    top_50_matches = []
    top_50_indices = sorted_indices[:50].cpu().numpy()
    for idx in top_50_indices:
        ref_name = reference_image_names[idx]
        ref_id = reference_image_ids[idx]
        distance = query_distance[idx].item()
        top_50_matches.append({
            'reference_image_name': ref_name,
            'reference_image_id': ref_id,
            'distance': distance
        })

    # Store the top 20 matches in the dictionary with the query image name as the key
    query_image_name = query_image_names[i]
    top_matches_data[query_image_name] = top_50_matches

if total_queries > 0:
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
else:
    print("No valid query images found.")

# --------------------- Save Top Matches --------------------- #
# Save the top matches data to a JSON file
with open('top_50_matches.json', 'w') as f:
    json.dump(top_matches_data, f, indent=4)

print("\nTop 50 matches for all query images saved to 'top_20_matches_all.json'.")
