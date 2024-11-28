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
    # Adjust the regex based on your image naming convention
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

batch_size = 64
num_workers = 4

query_dataset = ImageDataset(query_folder, transform=transform)
reference_dataset = ImageDataset(reference_folder, transform=transform)

query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
reference_loader = DataLoader(reference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

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

# Normalize embeddings
reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

# --------------------- Similarity Computation --------------------- #
# Use cosine similarity
similarities = torch.mm(query_embeddings, reference_embeddings.t())

# --------------------- Evaluation --------------------- #
correct_matches_top1 = 0
correct_matches_top5 = 0
correct_matches_top10 = 0
correct_matches_top20 = 0
correct_matches_top35 = 0
correct_matches_top50 = 0

total_queries = 0

for i, query_id in enumerate(query_image_ids):
    if query_id == -1:
        continue
    total_queries += 1
    query_similarity = similarities[i]
    sorted_indices = torch.argsort(query_similarity, descending=True)
    sorted_ids = [reference_image_ids[idx] for idx in sorted_indices.cpu().numpy()]
    # Find all indices where the ID matches
    relevant_indices = [idx for idx, ref_id in enumerate(sorted_ids) if ref_id == query_id]
    if not relevant_indices:
        continue  # No relevant items found
    first_relevant_rank = relevant_indices[0] + 1  # +1 because ranks start at 1
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
