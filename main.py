import cv2
import os
import numpy as np

def find_color(image_path, proximity_threshold):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in HSV
    lower_blue = np.array([100, 50, 50], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    # Create a mask using inRange to filter out pixels outside the blue color range
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Get the coordinates of non-zero pixels in the mask
    blue_points = np.column_stack(np.where(blue_mask > 0))
    
    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=blue_mask)

    # Group the coordinates based on proximity threshold
    grouped_coords = group_coordinates(blue_points, proximity_threshold)

    print(len(grouped_coords))

    # Display the original image and the result
    cv2.imshow('Original Image', image)
    cv2.imshow('Result Image (Blue)', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def group_coordinates(coordinates, proximity_threshold):
    grouped_points = []

    for coord in coordinates:
        # Check if the point can be added to an existing group
        added_to_group = False
        for group in grouped_points:
            for group_coord in group:
                distance = np.linalg.norm(coord - group_coord)
                if distance < proximity_threshold:
                    group.append(coord)
                    added_to_group = True
                    break

        # If the point couldn't be added to any existing group, create a new group
        if not added_to_group:
            grouped_points.append([coord])

    return grouped_points


if __name__ == "__main__":
    image_path = 'Input/Board4.jpg'
    proximity_threshold = 5
    find_color(image_path, proximity_threshold)
