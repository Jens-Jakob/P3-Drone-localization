import os
import shutil
import random

def select_random_images(src_folder, dest_folder, percentage=0.2):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # List all files in the source folder
    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # Calculate the number of images to select
    num_images_to_select = int(len(images) * percentage)

    # Randomly select images
    selected_images = random.sample(images, num_images_to_select)

    # Copy selected images to the destination folder
    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dest_path = os.path.join(dest_folder, image)
        shutil.copy(src_path, dest_path)

    print(f"Copied {num_images_to_select} images to {dest_folder}")

# Example usage
src_folder = '/Users/jens-jakobskotingerslev/Documents/GitHub/P3-QSS/Støvring_siamese/data/Støvring_UAV'
dest_folder = '/Users/jens-jakobskotingerslev/Documents/GitHub/P3-QSS/Støvring_siamese/data/Støvring_UAV_test'
select_random_images(src_folder, dest_folder)
