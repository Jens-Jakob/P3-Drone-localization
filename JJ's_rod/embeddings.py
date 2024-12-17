import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import time
from model import SiameseNetwork


# ---------------------- Helper Functions ---------------------- #
def compute_embeddings_in_batches(image_paths, model, device, transform, batch_size=32):
    embeddings = []
    image_names = []
    num_images = len(image_paths)
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_imgs = []
            batch_img_names = []
            for img_path in batch_paths:
                img_name = os.path.basename(img_path)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                batch_imgs.append(img)
                batch_img_names.append(img_name)
            batch_imgs = torch.stack(batch_imgs).to(device)
            outputs = model.forward_once(batch_imgs)
            embeddings.append(outputs.cpu())  # Move to CPU to save space if using GPU
            image_names.extend(batch_img_names)
            # Optional progress indicator
            print(f"Processed {i + len(batch_paths)}/{num_images} images.")
    embeddings = torch.cat(embeddings, dim=0)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_images
    print(f"Computed embeddings for {num_images} images in {total_time:.2f} seconds "
          f"({avg_time_per_image:.4f} seconds per image).")
    return embeddings, image_names


# ---------------------- Main Execution ---------------------- #
def main():
    parser = argparse.ArgumentParser(
        description='Compute and save embeddings for reference images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--reference_folder', type=str, required=True,
        help='Path to the folder containing reference satellite images.')
    parser.add_argument(
        '--siamese_checkpoint', type=str, required=True,
        help='Path to the Siamese network checkpoint.')
    parser.add_argument(
        '--output_path', type=str, default='reference_embeddings.pth',
        help='Path to save the computed embeddings.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for computing embeddings.')
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Use GPU for computation if available.')
    opt = parser.parse_args()

    total_start_time = time.time()  # Start timing the entire script

    # -------------------- Device and Model Setup -------------------- #
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Load Siamese Network
    siamese_model = SiameseNetwork().to(device)
    checkpoint = torch.load(opt.siamese_checkpoint, map_location=device)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    siamese_model.eval()

    # ---------------------- Image Transformations ---------------------- #
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Apply the same normalization used during training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --------------------- Compute Embeddings --------------------- #
    # Get list of reference image paths
    reference_folder = opt.reference_folder
    reference_image_paths = [
        os.path.join(reference_folder, f) for f in os.listdir(reference_folder)
        if f.lower().endswith('.png') and not f.startswith('._')
    ]
    print(f"Found {len(reference_image_paths)} reference images.")

    # Compute and save embeddings
    embeddings, image_names = compute_embeddings_in_batches(
        reference_image_paths, siamese_model, device, transform, batch_size=opt.batch_size)

    # Save embeddings and image names
    torch.save({'embeddings': embeddings, 'image_names': image_names}, opt.output_path)
    print(f"Saved precomputed embeddings to {opt.output_path}")

    total_end_time = time.time()  # End timing the entire script
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
