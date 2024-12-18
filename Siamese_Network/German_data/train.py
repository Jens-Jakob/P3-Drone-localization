from model import SiameseNetwork as Siamesenetwork
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from Data_loader import Triplet_Loader
import torch
from torch.utils.data import DataLoader
import os
from torch.nn import TripletMarginLoss
import wandb
import random


def train(save_interval=15, checkpoint_dir="checks"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries_folder = 'queries'
    reference_folder = 'reference_views'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees=10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print('Data augmentation complete!')

    dataset = Triplet_Loader(queries_folder, reference_folder, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=14)

    model = Siamesenetwork().to(device)
    criterion = TripletMarginLoss(margin=0.2, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=15)

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] completed with Average Loss: {epoch_loss:.4f}")

        if epoch % save_interval == 0:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_dir_path = os.path.join(script_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir_path, exist_ok=True)  # Ensure checkpoint directory exists
            checkpoint_path = os.path.join(checkpoint_dir_path, f"50_triplet_model_epoch_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        scheduler.step()

    print("Training completed.")

if __name__ == "__main__":
    train()
