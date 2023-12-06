import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def find_color(image_path, proximity_threshold):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in HSV
    # FIND BLUE COLOR
    # lower_color_range = np.array([100, 50, 50], dtype=np.uint8)
    # upper_color_range = np.array([140, 255, 255], dtype=np.uint8)

    # FIND ORANGE COLOR
    # lower_color_range = np.array([5, 50, 50], dtype=np.uint8)
    # upper_color_range = np.array([15, 255, 255], dtype=np.uint8)

    # FIND GREEN COLOR
    lower_color_range = np.array([40, 70, 70], dtype=np.uint8)
    upper_color_range = np.array([80, 255, 255], dtype=np.uint8)


    # Create a mask using inRange to filter out pixels outside the blue color range
    blue_mask = cv2.inRange(hsv_image, lower_color_range, upper_color_range)

    # Get the coordinates of non-zero pixels in the mask
    blue_points = np.column_stack(np.where(blue_mask > 0))
    
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=blue_mask)

    # Group the coordinates based on proximity threshold
    grouped_coords = group_coordinates(blue_points, proximity_threshold)

    for i, group in enumerate(grouped_coords):
        center = np.mean(group, axis=0)
        print(f"Center of Group {i + 1}: {center}")

    # Display the original image and the result
    cv2.imshow('Original Image', image)
    cv2.imshow('Result Image (Color)', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def group_coordinates(coordinates, proximity_threshold):
    # Apply DBSCAN to group coordinates based on proximity
    dbscan = DBSCAN(eps=proximity_threshold, min_samples=1)
    labels = dbscan.fit_predict(coordinates)

    # Initialize a dictionary to store grouped points
    grouped_points = {}

    # Iterate through the points and group them based on DBSCAN labels
    for i, label in enumerate(labels):
        if label not in grouped_points:
            grouped_points[label] = []
        grouped_points[label].append(coordinates[i])

    # Convert the dictionary values to a list of lists
    grouped_points_list = list(grouped_points.values())

    return grouped_points_list

if __name__ == "__main__":
    image_path = 'Input/Board1-3.jpg'
    proximity_threshold = 15
    find_color(image_path, proximity_threshold)
