import cv2
import numpy as np
from scipy.ndimage import label, generate_binary_structure

mask_path = '/workspace/baldr/baldr/Counter/imgs/trees_tanks.jpg'  
img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
print(f"img.shape: {img.shape}")
if img.ndim == 3:
    img = img[:, :, 0]  

_, binary_mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

assert binary_mask.ndim == 2, f"binary_mask is not 2D, shape: {binary_mask.shape}"

structure = generate_binary_structure(2, 2)

labeled_mask, num_objects = label(binary_mask, structure=structure)
print(f"Number of segmented objects: {num_objects}")

kernel = np.ones((7, 7), np.uint8)  

eroded_mask = cv2.erode(binary_mask, kernel, iterations=3)  

dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)  

cleaned_labeled_mask, cleaned_num_objects = label(dilated_mask, structure=structure)
print(f"Number of segmented objects after aggressive noise removal: {cleaned_num_objects}")

cv2.imshow("Cleaned Mask", dilated_mask * 255)  
cv2.waitKey(0)
cv2.destroyAllWindows()
