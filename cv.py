import cv2
import numpy as np
import os

# Define augmentation functions
def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def apply_laplacian(image, alpha=0.5):
    laplacian = cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    blended = cv2.addWeighted(image, 1 - alpha, laplacian, alpha, 0)  # Blend with original image
    return blended

def apply_gaussian_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_median_blur(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Dictionary of transformations
transformations = {
    "bright": adjust_brightness_contrast,
    "gaussian_noise": add_gaussian_noise,
    "laplacian": apply_laplacian,
    "gaussian_blur": apply_gaussian_blur,
    "median_blur": apply_median_blur,
    "bilateral": apply_bilateral_filter,
    "sharpen": apply_sharpening
}

# Function to process and save augmented videos in the SAME folder
def augment_video(video_path, output_folder, transformations):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

    base_name = os.path.basename(video_path).split('.')[0]

    # Apply all transformations and save in the same folder
    for name, transform in transformations.items():
        output_video_path = os.path.join(output_folder, f"{base_name}_{name}.mp4")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame position
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            transformed_frame = transform(frame)
            writer.write(transformed_frame)

        writer.release()
        print(f" Saved: {output_video_path}")

    cap.release()

# Paths to dataset folders 
dataset_root = r"C:/Users/Rocky/Desktop/dataset/barbel curl"
categories = ["barbel curl correct", "barbel curl incorrect"]

for category in categories:
    input_folder = os.path.join(dataset_root, category)

    # Debugging: Print folder path
    print(f" Checking folder: {input_folder}")

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"⚠ Warning: Input folder '{input_folder}' not found. Please check the path.")
        continue

    for video_file in os.listdir(input_folder):
        if video_file.endswith(".mp4"):  
            video_path = os.path.join(input_folder, video_file)
            augment_video(video_path, input_folder, transformations)  

print("Augmentation complete! Augmented videos saved in the original folders.")
