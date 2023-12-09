import cv2
import os
import numpy as np
import random
import math
BULLSEYE_COORDS = (361, 412)

def sift_algorithm(image1):
    # Implement the SIFT algorithm to extract features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image1, None)

    return keypoints, descriptors

def score_system(point):
    distance = math.sqrt((point[0] - BULLSEYE_COORDS[0])**2 + (point[1] - BULLSEYE_COORDS[1])**2)
    if distance >300:
        return 0
    elif distance <11:
        return 10
    elif distance <36:
        return 9
    elif distance <66:
        return 8
    elif distance <91:
        return 7
    elif distance <126:
        return 6
    elif distance <151:
        return 5
    elif distance <186:
        return 4
    elif distance <211:
        return 3
    elif distance <241:
        return 2
    else:
        return 1

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
    image_path = 'TestPhoto.png'
    image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.circle(image1, (BULLSEYE_COORDS[0]+10, BULLSEYE_COORDS[1]), 2, 255, -1)
  
    # cv2.imshow('Warped Image', image1)
    # cv2.waitKey(0)
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
            warped_image = cv2.warpPerspective(image2, homography, (image2.shape[1]+100, image2.shape[0]+100))

            # Display the result
            # result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow('Brute Force Matching', result_image)
            # text = f'Image Name: {filename}'
            # cv2.putText(warped_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_coordinate = random.randint(BULLSEYE_COORDS[0]-300, BULLSEYE_COORDS[0]+300 - 1)
            y_coordinate = random.randint(BULLSEYE_COORDS[1]-300, BULLSEYE_COORDS[1]+300 - 1)

            point_color = 255
            point_radius = 5
            text = f' Score : {score_system((x_coordinate,y_coordinate))}'
            cv2.putText(warped_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(warped_image, (x_coordinate,y_coordinate), point_radius, point_color, -1)
            cv2.imshow('Warped Image', warped_image)

            cv2.waitKey(0)

        cv2.destroyAllWindows()

