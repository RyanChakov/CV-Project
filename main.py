import cv2
import os
import numpy as np

def preprocess_images(input_folder, output_folder):
    preprocessed_images = []
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            # output_path = os.path.join(output_folder, filename)

            # Preprocess the image (standardize format, resolution, lighting conditions)
            # Add your preprocessing code here
            
            # Save the preprocessed image to the output folder
            # cv2.imwrite(output_path, preprocessed_image)
    return preprocessed_images

def sift_algorithm(image1):
    # Implement the SIFT algorithm to extract features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image1, None)
    # image_with_keypoints = cv2.drawKeypoints(image1, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image with keypoints
    # cv2.imshow('SIFT Keypoints', image_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return keypoints, descriptors

def orb_algorithm(image):
    # Create ORB object
    orb = cv2.ORB_create()

    # Detect and compute keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def matching_algorithm(descriptors_database, descriptors_input):
    # Implement a matching algorithm to compare input image descriptors with the database
    # Add your matching algorithm code here
    # Return the best match and its score
    return 0

def score_system(match):
    # Implement a scoring system based on the matched features
    # Return the score for the dart on the board
    return 0

def dart_recognition(input_folder, output_folder, database_path):
    # Preprocess images in the input folder
    preprocess_images(input_folder, output_folder)

    # Load the database of known dart patterns
    database_images = []
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(database_path, filename)
            database_images.append(cv2.imread(image_path))

    # Loop through each preprocessed image in the output folder
    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(output_folder, filename)
            input_image = cv2.imread(input_path)

            # Extract SIFT features from the input image
            keypoints_input, descriptors_input = sift_algorithm(input_image)

            # Loop through each image in the database and find the best match
            best_match = None
            best_score = 0
            for database_image in database_images:
                keypoints_database, descriptors_database = sift_algorithm(database_image)

                # Compare descriptors and get a match
                match = matching_algorithm(descriptors_database, descriptors_input)

                # Update best match if the current match has a higher score
                if match['score'] > best_score:
                    best_score = match['score']
                    best_match = match

            # Determine the score for the dart on the board
            score = score_system(best_match)

            # Display or save the results
            # Add your code to display or save the results here

def brute_force_matching(descriptors1, descriptors2):
    # Create a brute-force matcher object
    bf = cv2.BFMatcher()

    # Perform knn matching
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    return good_matches

if __name__ == "__main__":
     # Example image path
    image_path = 'FullBoard.png'
    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = sift_algorithm(image1)

    for filename in os.listdir("Input/"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join("Input/", filename)

            # Read the input image
            image2 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Apply SIFT algorithm to the input image
            keypoints2, descriptors2 = sift_algorithm(image2)

            # Perform brute-force matching
            matches = brute_force_matching(descriptors1, descriptors2)

            # Find homography
            pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Use RANSAC to find the homography matrix
            homography, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)

            # Warp the input image
            warped_image = cv2.warpPerspective(image2, homography, (image2.shape[1], image2.shape[0]))

            # Display the result
            result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Brute Force Matching', result_image)
            text = f'Image Name: {filename}'
            cv2.putText(warped_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Warped Image', warped_image)

            cv2.waitKey(0)

        cv2.destroyAllWindows()

