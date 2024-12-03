from pathlib import Path
import argparse
import numpy as np
import torch
import pandas as pd
import json  # Import json module to read the JSON file

from models.matching import Matching
from models.utils import read_image, AverageTimer

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image matching and similarity evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Arguments
    parser.add_argument(
        '--json_file', type=str, default="clean_fixet.json",
        help='Path to the JSON file containing image pairs to compare')
    parser.add_argument(
        '--drone_image_dir', type=str, default="queries/",
        help='Path to the directory containing drone images')
    parser.add_argument(
        '--satellite_image_dir', type=str, default="reference_views/",
        help='Path to the directory containing satellite images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Directory where results and visualization images will be saved')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input images before processing. '
             'If two numbers, resize to that exact size. '
             'If one number, resize the max dimension. '
             'If -1, do not resize.')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after converting uint8 to float')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights to use')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by SuperPoint '
             '(-1 keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detection confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non-Maximum Suppression (NMS) radius')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations in SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue matching threshold')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize matches and save the plots')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot keypoints along with matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='File extension for visualization images')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Display visualization images with OpenCV before saving')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force computation on CPU')

    opt = parser.parse_args()
    print(opt)

    # Load SuperPoint and SuperGlue models
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print(f'Running inference on device: "{device}"')

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create output directory if it doesn't exist
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    timer = AverageTimer(newline=True)

    # Load the JSON file
    with open(opt.json_file, 'r') as f:
        image_pairs = json.load(f)

    results = []

    # Iterate over each drone image and its list of satellite images
    for drone_image_name, satellite_image_names in image_pairs.items():
        drone_image_path = Path(opt.drone_image_dir) / drone_image_name
        stem0 = drone_image_path.stem

        # Load the drone image
        image0, inp0, scales0 = read_image(
            drone_image_path, device, opt.resize, 0, opt.resize_float)
        if image0 is None:
            print(f'Problem reading drone image: {drone_image_path}')
            continue

        best_match = None
        best_num_matches = -1
        best_avg_confidence = -1
        best_satellite_image = None

        for satellite_image_name in satellite_image_names:
            satellite_image_path = Path(opt.satellite_image_dir) / satellite_image_name
            stem1 = satellite_image_path.stem

            # Load the satellite image
            image1, inp1, scales1 = read_image(
                satellite_image_path, device, opt.resize, 0, opt.resize_float)
            if image1 is None:
                print(f'Problem reading satellite image: {satellite_image_path}')
                continue

            # Perform matching
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Extract matching keypoints
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            # Compute similarity metrics
            num_matches = len(mkpts0)
            avg_confidence = np.mean(mconf) if num_matches > 0 else 0
            print(
                f'Comparing drone image {stem0} against satellite image {stem1}: '
                f'Number of matches: {num_matches}, Average match confidence: {avg_confidence:.4f}')

            # Update best match if current match is better
            if num_matches > best_num_matches or (
                    num_matches == best_num_matches and avg_confidence > best_avg_confidence):
                best_num_matches = num_matches
                best_avg_confidence = avg_confidence
                best_satellite_image = stem1

        # Store best match for this drone image
        if best_satellite_image is not None:
            results.append({
                'drone_image': stem0,
                'best_match_satellite_image': best_satellite_image,
                'num_matches': best_num_matches,
                'avg_confidence': best_avg_confidence
            })

    # Save best matches to CSV
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='drone_image', inplace=True)
    results_df.to_csv(output_dir / 'final_matchglue.csv', index=False)
    print(f"Saved best matching results to {output_dir / 'best_matching_results.csv'}")
    print("All done!")
