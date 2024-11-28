from model import SiameseNetwork as Siamesenetwork
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from data_loading_triplet import TripletDatasetFromCSV
import torch
from torch.utils.data import DataLoader
import os
from torch.nn import TripletMarginLoss
import wandb

# Start a new wandb run to track this script
wandb.init(
    project="Siamesetest_støvring_data",
    config={
        "learning_rate": 0.01,
        "architecture": "Siamese Network with ResNet101",
        "dataset": "Støvring Data",
        "epochs": 20,
        "scheduler": "ReduceLROnPlateau",
        "optimizer": "SGD",
        "batch_size": 32,
    }
)

print('Starting up')

def train(save_interval=5, checkpoint_dir="checks"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your data
    csv_file = '../data/triplet_data.csv'  # Update this to the actual path
    uav_image_root = 'Støvring_UAV'     # Update to the actual path
    satellite_image_root = 'Støvring_satellite'  # Update to the actual path

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

    # Use the updated data loader
    dataset = TripletDatasetFromCSV(csv_file, uav_image_root, satellite_image_root, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Siamesenetwork().to(device)
    criterion = TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    num_epochs = 11
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

        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] completed with Average Loss: {epoch_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save checkpoint if required
        if epoch % save_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"triplet_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # Step the scheduler with the current epoch loss
        scheduler.step(epoch_loss)

    print("Training completed.")
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
