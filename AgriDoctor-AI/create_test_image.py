import cv2
import numpy as np

# Create a blank green background
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = (40, 150, 40)

# Draw a simple leaf shape
points = np.array([
    [150, 40],
    [220, 120],
    [150, 260],
    [80, 120]
], dtype=np.int32)

cv2.fillPoly(image, [points], (50, 200, 70))

# Draw the leaf vein
cv2.line(image, (150, 40), (150, 260), (255, 255, 255), 2)

# Save the image
cv2.imwrite("test_leaf.jpg", image)

print("Sample leaf image created successfully!")