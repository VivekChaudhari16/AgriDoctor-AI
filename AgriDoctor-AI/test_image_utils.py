from src.image_utils import load_image

image_path = "test_leaf.jpg"

try:
    image = load_image(image_path)
    print("Image loaded successfully!")
    print("Image dimensions:", image.shape)

except ValueError as error:
    print(error)