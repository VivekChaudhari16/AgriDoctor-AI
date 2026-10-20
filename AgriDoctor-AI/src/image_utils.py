import cv2


def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image