import cv2
import os
import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering
BULLSEYE_COORDS = (361, 412)

def find_color(input_path):
    image = cv2.imread(input_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color_range = np.array([40, 70, 70], dtype=np.uint8)
    upper_color_range = np.array([80, 255, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv_image, lower_color_range, upper_color_range)

    blue_points = np.column_stack(np.where(blue_mask > 0))

    np.random.seed(42)

    # Use hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='complete')

    # Fit the model and get cluster labels
    labels = clustering.fit_predict(blue_points)

    # Calculate cluster centers based on the mean of points in each cluster
    unique_labels = np.unique(labels)
    cluster_centers = [np.mean(blue_points[labels == label], axis=0) for label in unique_labels]
    # print(cluster_centers)
    return cluster_centers

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
    elif distance <129:
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

    image_path = 'StraightBoard.png'
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
            pts1 = cv2.convertPointsToHomogeneous(np.array([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2))
            pts2 = cv2.convertPointsToHomogeneous(np.array([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2))

            homography, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)

            warped_image = cv2.warpPerspective(image2, homography, (image2.shape[1]+100, image2.shape[0]+100))
            points = find_color(input_path)

            score=0
            for point in points:
                point_homogeneous = [point[1],point[0], 1]
                transformed_point = np.dot(homography, point_homogeneous)
                new_x = transformed_point[0] / transformed_point[2]
                new_y = transformed_point[1] / transformed_point[2]
                score=score+score_system([new_x,new_y])
                cv2.circle(warped_image, (int(new_x),int(new_y)), 5, 255, -1)

            text = f' Score : {score}'
            cv2.putText(warped_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Warped Image', warped_image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()