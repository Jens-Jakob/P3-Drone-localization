import cv2
import numpy as np
import os

def ORB_matching(query_dir, ref_dir):
    Best_q_match = []  # List to store best matches for all query images
    Correct = []  # List to store correct matches

    # Preload and preprocess reference images
    ref_images_data = []
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    orb = cv2.ORB_create()

    for img_name in os.listdir(ref_dir):
        img_full_path = os.path.join(ref_dir, img_name)
        img2 = cv2.imread(img_full_path)
        
        if img2 is None:
            print(f"Error loading reference image: {img_name}")
            continue
            
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.filter2D(img2_gray, -1, kernel)

        # Find keypoints and descriptors with ORB for the reference image
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
        
        if descriptors2 is not None:
            ref_images_data.append((img_name, keypoints2, descriptors2))

    # Loop through all query images
    for query_name in os.listdir(query_dir):
        img1_path = os.path.join(query_dir, query_name)
        img1 = cv2.imread(img1_path)
        
        if img1 is None:
            print(f"Error loading query image: {query_name}")
            continue

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.filter2D(img1_gray, -1, kernel)

        # Find keypoints and descriptors with ORB for the query image
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
        
        if descriptors1 is None:
            print(f"No descriptors found in query image: {query_name}")
            continue

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        Img_distance_match = []  # List to store distance matches for this query image

        # Match the query image with each preloaded reference image
        for img_name, keypoints2, descriptors2 in ref_images_data:
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            # If matches exist, process the first match
            if matches:
                first_match = matches[0]
                print(f"Distance of the first match between {query_name} and {img_name}: {first_match.distance}")
                Img_distance_match.append((query_name, img_name, first_match.distance))

        # Find the best match (smallest distance) for the current query image
        if Img_distance_match:
            best_match_for_query = min(Img_distance_match, key=lambda x: x[2])  # Sort by distance
            Best_q_match.append(best_match_for_query)

            if best_match_for_query[0] in best_match_for_query[1]:  # Example condition for correctness
                Correct.append(best_match_for_query)

    return Best_q_match, Correct

query_dir = ""
ref_dir = ""

best_matches, correct_matches = ORB_matching(query_dir, ref_dir)

print(f"Number of best matches found: {len(best_matches)}")
for query_img, ref_img, distance in best_matches:
    print(f"Best match for query '{query_img}': Reference image '{ref_img}' with distance {distance}")

print(f"Total number of correct matches: {len(correct_matches)}")
