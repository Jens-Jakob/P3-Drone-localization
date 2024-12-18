import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import json
import numpy as np
import pandas as pd
import argparse
import time
from torch.utils.data import Dataset, DataLoader
from model import SiameseNetwork
from matching import Matching
from utils import read_image, AverageTimer

def extract_id(image_name):
    match = re.search(r'(\d+)', image_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def compute_embeddings(image_paths, model, device, transform):
    embeddings = []
    image_names = []
    image_ids = []
    start_time = time.time()
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            output = model.forward_once(img)
            embeddings.append(output.cpu())
            image_names.append(img_name)
            image_ids.append(extract_id(img_name))
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images.")
    embeddings = torch.cat(embeddings, dim=0)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(image_paths)
    print(f"Computed embeddings for {len(image_paths)} images in {total_time:.2f} seconds "
          f"({avg_time_per_image:.4f} seconds per image).")
    return embeddings, image_names, image_ids

def load_reference_embeddings(embeddings_path, device):
    data = torch.load(embeddings_path, map_location=device)
    embeddings = data['embeddings'].to(device)
    image_names = data['image_names']
    return embeddings, image_names

def perform_superglue_matching(query_image_path, reference_image_paths, device, superglue_config):
    matching = Matching(superglue_config).eval().to(device)
    start_time = time.time()
    image0, inp0, scales0 = read_image(
        query_image_path, device, [-1], 0, False)
    if image0 is None:
        print(f'Problem reading query image: {query_image_path}')
        return None, -1, -1, []
    best_match = None
    best_num_matches = -1
    best_avg_confidence = -1
    best_satellite_image = None
    results = []
    total_matching_time = 0
    for idx, ref_image_path in enumerate(reference_image_paths):
        image1, inp1, scales1 = read_image(
            ref_image_path, device, [-1], 0, False)
        if image1 is None:
            print(f'Problem reading reference image: {ref_image_path}')
            continue
        match_start_time = time.time()
        with torch.no_grad():
            pred = matching({'image0': inp0, 'image1': inp1})
        match_end_time = time.time()
        total_matching_time += match_end_time - match_start_time
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        num_matches = len(mkpts0)
        avg_confidence = np.mean(mconf) if num_matches > 0 else 0
        print(
            f'Comparing query image {os.path.basename(query_image_path)} against reference image {os.path.basename(ref_image_path)}: '
            f'Number of matches: {num_matches}, Average match confidence: {avg_confidence:.4f}')
        if num_matches > best_num_matches or (
                num_matches == best_num_matches and avg_confidence > best_avg_confidence):
            best_num_matches = num_matches
            best_avg_confidence = avg_confidence
            best_satellite_image = ref_image_path
        results.append({
            'drone_image': os.path.basename(query_image_path),
            'satellite_image': os.path.basename(ref_image_path),
            'num_matches': num_matches,
            'avg_confidence': avg_confidence
        })
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / len(reference_image_paths)
    print(f"SuperGlue matching for {len(reference_image_paths)} images took {total_time:.2f} seconds.")
    print(f"Average matching time per image: {avg_time_per_image:.4f} seconds.")
    print(f"Total time spent in matching (excluding I/O): {total_matching_time:.2f} seconds.")
    return best_satellite_image, best_num_matches, best_avg_confidence, results

def get_location(image_name, poses_csv_path):
    filename_to_find = image_name
    if not filename_to_find.endswith('.png'):
        filename_to_find += '.png'
    df = pd.read_csv(poses_csv_path)
    row = df.loc[df['filename'] == filename_to_find]
    if not row.empty:
        lat = row.iloc[0]['lat']
        lon = row.iloc[0]['lon']
        return lat, lon
    else:
        print(f"Filename {filename_to_find} not found in {poses_csv_path}.")
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description='Process a single image to find its location.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_image', type=str, required=True,
        help='Path to the input drone image.')
    parser.add_argument(
        '--reference_folder', type=str, default='/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/reference_views2',
        help='Path to the folder containing reference satellite images.')
    parser.add_argument(
        '--poses_csv', type=str, default='/Users/jens-jakobskotingerslev/Documents/GitHub/P3-Drone-localization/Siamese_Network/German_data/poses.csv',
        help='Path to the CSV file containing image poses (latitude and longitude).')
    parser.add_argument(
        '--siamese_checkpoint', type=str, default='/Users/jens-jakobskotingerslev/Documents/GitHub/P3-Drone-localization/Siamese_Network/German_data/checkpoints/50_triplet_model_epoch_epoch_15.pth',
        help='Path to the Siamese network checkpoint.')
    parser.add_argument(
        '--superglue_weights', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights to use.')
    parser.add_argument(
        '--embeddings_path', type=str, default='/Users/jens-jakobskotingerslev/Documents/GitHub/P3-Drone-localization/Siamese_Network/German_data/reference_embeddings.pth',
        help='Path to the precomputed reference embeddings.')
    opt = parser.parse_args()
    MIN_NUM_MATCHES = 100
    MIN_AVG_CONFIDENCE = 0.2
    total_start_time = time.time()
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    siamese_model = SiameseNetwork().to(device)
    checkpoint = torch.load(opt.siamese_checkpoint, map_location=device)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    siamese_model.eval()
    superglue_config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        },
        'superglue': {
            'weights': opt.superglue_weights,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    end_time = time.time()
    print(f"Device and Model Setup took {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    end_time = time.time()
    print(f"Image Transformations setup took {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    query_image_path = opt.input_image
    query_embeddings, query_image_names, query_image_ids = compute_embeddings(
        [query_image_path], siamese_model, device, transform)
    end_time = time.time()
    print(f"Computed embeddings for the input image in {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    reference_embeddings, reference_image_names = load_reference_embeddings(opt.embeddings_path, device)
    end_time = time.time()
    print(f"Loaded precomputed embeddings in {end_time - start_time:.2f} seconds.")
    reference_folder = opt.reference_folder
    reference_image_paths = [os.path.join(reference_folder, name) for name in reference_image_names]
    start_time = time.time()
    distances = torch.cdist(query_embeddings, reference_embeddings, p=2)
    distances = distances[0]
    end_time = time.time()
    print(f"Similarity computation took {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    top_k = 10
    sorted_indices = torch.argsort(distances)[:top_k]
    top_reference_image_paths = [reference_image_paths[idx] for idx in sorted_indices.cpu().numpy()]
    end_time = time.time()
    print(f"Selecting top {top_k} matches took {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    best_satellite_image, best_num_matches, best_avg_confidence, matching_results = perform_superglue_matching(
        query_image_path, top_reference_image_paths, device, superglue_config)
    end_time = time.time()
    print(f"SuperGlue matching took {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    if best_satellite_image:
        if best_num_matches >= MIN_NUM_MATCHES and best_avg_confidence >= MIN_AVG_CONFIDENCE:
            best_image_name = os.path.basename(best_satellite_image)
            print(f"Best matching image: {best_image_name}")
            lat, lon = get_location(best_image_name, opt.poses_csv)
            if lat is not None and lon is not None:
                print(f"Latitude: {lat}, Longitude: {lon}")
            else:
                print("Coordinates not found in the poses CSV.")
        else:
            print("I don't know the current location.")
    else:
        print("No matching satellite image found.")
    end_time = time.time()
    print(f"Getting location took {end_time - start_time:.2f} seconds.")
    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
