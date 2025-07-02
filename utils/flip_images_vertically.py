import os
from PIL import Image


def flip_images_vertically(folder_path):
    """Vertically flips all images in the specified folder and replaces the originals."""
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Supported image formats
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and any(
            filename.lower().endswith(ext) for ext in image_extensions
        ):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Flip the image vertically
                    # flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    flipped_img = img.transpose(
                        Image.FLIP_LEFT_RIGHT
                    )  # horizontal flip
                    # Save the flipped image, replacing the original
                    flipped_img.save(file_path)
                    print(f"Flipped: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("All images have been processed.")


# Example usage
# Replace with your actual folder path
folder_path = "C:\\Users\\Personal\\OneDrive - National University of Sciences & Technology\\University\\FYP\\Data\\dataset_v2\\dataset_v2"
flip_images_vertically(folder_path)
