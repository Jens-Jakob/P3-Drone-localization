import os
from feature_extractor import extract_features, resnet_feature_extractor
from image_loader import load_and_preprocess_image, data_loader2
from similarity_calculator import calculate_similarity
from utils import get_drone_images, get_satellite_images

# Paths to data folders
drone_folder_path = r"drone_images"  # Folder containing drone images
satellite_folder_path = r"reference_views"  # Folder containing satellite images


# top_n = amount of matches printed (in image_loader.py)
def main(drone_folder_path, satellite_folder_path, top_n=3, max_images=2706):
    # Initialize neural network
    model = resnet_feature_extractor()
    # Load the selected amount of images using data_loader2
    drone_image_list = get_drone_images(drone_folder_path)
    satellite_image_list = get_satellite_images(satellite_folder_path)

    # Get the filtered list of images
    drone_images = data_loader2(drone_image_list, max_images)
    satellite_images = data_loader2(satellite_image_list, max_images)

    # Empty list to store satellite features
    satellite_features = []

    # Extract features for all satellite images
    for sat_img in satellite_images:
        try:
            # Convert images to tensors and extract features
            sat_img_tensor = load_and_preprocess_image(sat_img)
            features = extract_features(model, sat_img_tensor)
            satellite_features.append(features)
        except Exception as e:
            print(f"Skipping unidentifiable image: {sat_img} due to error: {e}")

    # Initialize counters and results storage
    match_count = 0
    second_match_count = 0
    third_match_count = 0
    top_20_count = 0  # New counter for matches in top 20
    total_checked = 0
    results = []

    # Compare each drone image to the satellite images
    for drone_img in drone_images:
        total_checked += 1

        try:
            print(f"\nMatching for Drone Image: {os.path.basename(drone_img)}")
            # Convert and extract features from the drone image
            drone_img_tensor = load_and_preprocess_image(drone_img)
            drone_features = extract_features(model, drone_img_tensor)

            # Calculate similarities with all satellite images
            similarity = calculate_similarity(drone_features, satellite_features)
            image_similar_pairs = list(zip(satellite_images, similarity))
            image_similar_pairs.sort(key=lambda x: x[1], reverse=True)

            # Print top N matches for this drone image
            print(f"Top {top_n} matches for {os.path.basename(drone_img)}:")
            correct_in_top_20 = False  # Flag for correct match within top 20

            for i, (sat_img, similarity) in enumerate(image_similar_pairs[:max(top_n, 20)]):  # Check top N and up to 20
                sat_base_name = os.path.splitext(os.path.basename(sat_img))[0]

                if os.path.splitext(os.path.basename(drone_img))[0] == sat_base_name:
                    match_count += 1
                    correct_in_top_20 = True  # Correct match found within top 20
                    if i == 1:
                        second_match_count += 1
                    elif i == 2:
                        third_match_count += 1

                if i < top_n:
                    print(f"{i + 1}: Satellite Image: {os.path.basename(sat_img)}, Similarity: {similarity:.4f}")
                    results.append((os.path.basename(drone_img), os.path.basename(sat_img), similarity))

            # Update top 20 count if a correct match was found
            if correct_in_top_20:
                top_20_count += 1

            # Print the best match (highest similarity)
            best_match_image = image_similar_pairs[0][0]
            print(
                f"\nBest match for {os.path.basename(drone_img)}: {os.path.basename(best_match_image)} with similarity: {image_similar_pairs[0][1]:.4f}")
            print("\n" + "-" * 40)  # Separator line

        except Exception as e:
            print(f"Skipping unidentifiable drone image: {drone_img} due to error: {e}")

    # Print total matches found and percentage
    if total_checked > 0:
        print(f"\nTotal perfect matches found: {match_count} out of {total_checked} checked images.")
        print(f"Percentage of perfect matches: {(match_count / total_checked) * 100:.2f}%")
        print(f"Percentage of correct match in second position: {(second_match_count / total_checked) * 100:.2f}%")
        print(f"Percentage of correct match in third position: {(third_match_count / total_checked) * 100:.2f}%")
        print(f"Percentage of correct match within top 20: {(top_20_count / total_checked) * 100:.2f}%")


# Main function call
if __name__ == "__main__":
    main(drone_folder_path, satellite_folder_path)
