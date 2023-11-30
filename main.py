import cv2
import os
import numpy as np

def find_color(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue color in HSV
    lower_blue = np.array([100, 50, 50], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    # Create a mask using inRange to filter out pixels outside the blue color range
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=blue_mask)

    # Display the original image and the result
    cv2.imshow('Original Image', image)
    cv2.imshow('Result Image (Blue)', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'Input/Board3.jpg'
    # Example usage
    find_color(image_path)
