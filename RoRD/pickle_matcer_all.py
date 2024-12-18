import argparse
import os
import time
import csv
import cv2
import numpy as np
import pydegensac
import torch
from PIL import Image
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image
import pickle
import signal
import sys

# Update the save_cache_on_exit function to save both caches
def save_cache_on_exit(signal, frame):
    save_features_cache(features_cache1, args.cache_file1)
    save_features_cache(features_cache2, args.cache_file2)
    print("\nCaches saved before exit.")
    sys.exit(0)

# Update signal handlers
signal.signal(signal.SIGINT, save_cache_on_exit)
signal.signal(signal.SIGTERM, save_cache_on_exit)

parser = argparse.ArgumentParser(description='Feature extraction and matching script for folders')
parser.add_argument('--folder1', type=str, help="Path to the first folder of images", default="drone_images/")
parser.add_argument('--folder2', type=str, help="Path to the second folder of images", default="satellitefo/")
parser.add_argument('--preprocessing', type=str, default='caffe', help='Image preprocessing (caffe or torch)')
parser.add_argument('--model_file', type=str, help='Path to the full model')
parser.add_argument('--no-relu', dest='use_relu', action='store_false', help='Remove ReLU after the dense feature extraction module')
parser.add_argument('--output_csv', type=str, default='RoRD_all.csv', help='Path to save the best matches CSV file')
parser.add_argument('--cache_file1', type=str, default='cache_folderfu.pkl', help="Path to the features cache file for folder1")
parser.add_argument('--cache_file2', type=str, default='cache_folderfu2.pkl', help="Path to the features cache file for folder2")

# Function to save features cache to a file
def save_features_cache(cache, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to {cache_file}")


def extract(image, args, model, device):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)
    input_image = preprocess_image(image, preprocessing=args.preprocessing)
    with torch.no_grad():
        keypoints, scores, descriptors = process_multiscale(
            torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=device),
            model, scales=[1]
        )
    keypoints = keypoints[:, [1, 0, 2]]
    return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}

def rordMatching(feat1, feat2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(feat1['descriptors'], feat2['descriptors'])
    matches = sorted(matches, key=lambda x: x.distance)
    match1 = [m.queryIdx for m in matches]
    match2 = [m.trainIdx for m in matches]
    keypoints_left = feat1['keypoints'][match1, :2]
    keypoints_right = feat2['keypoints'][match2, :2]
    np.random.seed(42)

    if len(keypoints_left) >= 4 and len(keypoints_right) >= 4:
        try:
            _, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)
            return np.sum(inliers) if inliers is not None else 0
        except np.linalg.LinAlgError:
            print("Singular matrix encountered; skipping homography computation.")
            return 0
    else:
        print("Insufficient matches to compute homography.")
        return 0


def load_features_cache(cache_file):
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Cache file not found or is empty. Initializing a new cache.")
        return {}

if __name__ == '__main__':
    start_time = time.time()

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = D2Net(model_file=args.model_file, use_relu=args.use_relu, use_cuda=torch.cuda.is_available())
    print("Model loaded successfully.")

    # Initialize separate caches for each folder
    features_cache1 = load_features_cache(args.cache_file1)
    features_cache2 = load_features_cache(args.cache_file2)
    cache1_updated = False
    cache2_updated = False

    # Prepare CSV file to save best matches
    with open(args.output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image1', 'Best_Match_Image2', 'Inliers'])

        # Process images from folder1
        for img1_name in os.listdir(args.folder1):
            img1_path = os.path.join(args.folder1, img1_name)
            if not os.path.isfile(img1_path):
                continue

            print(f"\nProcessing image: {img1_name}")

            # Read and extract features for image1
            if img1_name not in features_cache1:
                image1 = np.array(Image.open(img1_path))
                feat1 = extract(image1, args, model, device)
                features_cache1[img1_name] = feat1
                cache1_updated = True
            else:
                feat1 = features_cache1[img1_name]

            best_match = None
            max_inliers = 0

            # Compare image1 with all images in folder2
            for img2_name in os.listdir(args.folder2):
                img2_path = os.path.join(args.folder2, img2_name)
                if not os.path.isfile(img2_path):
                    continue

                print(f"  Comparing with image: {img2_name}")

                # Read and extract features for image2
                if img2_name not in features_cache2:
                    image2 = np.array(Image.open(img2_path))
                    feat2 = extract(image2, args, model, device)
                    features_cache2[img2_name] = feat2
                    cache2_updated = True
                else:
                    feat2 = features_cache2[img2_name]

                # Perform matching and count inliers
                n_inliers = rordMatching(feat1, feat2)
                print(f"  Inliers found: {n_inliers}")

                if n_inliers > max_inliers:
                    max_inliers = n_inliers
                    best_match = img2_name

            # Write the best match to the CSV file
            csv_writer.writerow([img1_name, best_match, max_inliers])
            print(f"Best match for {img1_name}: {best_match} with {max_inliers} inliers")

    # Save caches if they were updated
    if cache1_updated:
        save_features_cache(features_cache1, args.cache_file1)
        print(f"Cache for folder1 updated and saved to {args.cache_file1}.")
    if cache2_updated:
        save_features_cache(features_cache2, args.cache_file2)
        print(f"Cache for folder2 updated and saved to {args.cache_file2}.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nAll matching results saved to {args.output_csv}.")
    print(f"Total execution time: {elapsed_time:.2f} seconds.")