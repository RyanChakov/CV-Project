import cv2
import os

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
    image_path = 'Union.png'

    # Read the image
    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('SIFT Keypoints', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image2_path = 'Input/Board2.jpg'

    # Read the image
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('SIFT Keypoints', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Call the SIFT algorithm
    keypoints, descriptors = sift_algorithm(image1)

    # Loop through each image in the input folder
    for filename in os.listdir("Input/"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join("Input/", filename)

            # Read the image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Apply SIFT algorithm
            keypoints2, descriptors2 = sift_algorithm(image)
            matches = brute_force_matching(descriptors,descriptors2)

            result_image = cv2.drawMatches(image1, keypoints, image, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Brute Force Matching', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Visualize the keypoints on the image
            # image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Save the result
            # cv2.imwrite(output_path, image_with_keypoints)
    # keypoints2, descriptors2 = sift_algorithm(image2)

    # matches = brute_force_matching(descriptors,descriptors2)

    # result_image = cv2.drawMatches(image1, keypoints, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result



