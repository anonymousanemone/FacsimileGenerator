import os
import uuid
import numpy as np
import cv2
from utils import read_image, save_image_cv
import matplotlib.pyplot as plt
from glob import glob


def denoise_image(img, sigma=6.0):
    k = int(np.ceil(2 * np.pi * sigma)) // 2 * 2 + 1
    denoised = cv2.GaussianBlur(img, (k, k), sigma)
    return denoised

def binarize(img_gray, method="otsu", thresh=128, adapt_win=11):
    if method == "otsu":
        _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "fixed":
        _, bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, adapt_win, 2)
    else:
        raise ValueError("Unknown binarization method")
    return bw

def detect_edges(img_gray, low=50, high=150):
    return cv2.Canny(img_gray, low, high)

"""
Tests for boundary and edge detection functions in processing.py.
Run with:
    python test_processing.py
"""

def test_boundary_detection_contours(img):
    """Test contour-based segmentation (boundary detection)."""
    print("[TEST] Boundary detection via contours")

    # mask = segment_by_contour(img, min_area=200)
    mask = img

    # Draw contours on original image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(outline, contours, -1, (0, 0, 255), 2)

    print(f"  Found {len(contours)} contour(s)")
    assert len(contours) > 0, "No contours found — boundary detection failed."

    # Display
    cv2.imshow("Original", img)
    cv2.imshow("Boundary mask", mask)
    cv2.imshow("Outlined object(s)", outline)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_contour(img):
    """
    Test contour-based segmentation (boundary detection).

    Updated:
    - selects the single largest contour
    - fills its interior into a mask
    - displays only that contour
    - returns (largest_contour, filled_mask)
    """
    print("[TEST] Boundary detection via contours")

    # Get mask from contour segmentation
    # mask = segment_by_contour(img, min_area=200)
    mask = img

    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Found {len(contours)} contour(s)")
    assert len(contours) > 0, "No contours found — boundary detection failed."

    # --- SELECT LARGEST CONTOUR ---
    largest_contour = max(contours, key=cv2.contourArea)

    # --- CREATE FILLED MASK OF LARGEST CONTOUR ---
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # --- OUTLINE ON ORIGINAL IMAGE ---
    if len(img.shape) == 2:
        outline = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        outline = img.copy()

    cv2.drawContours(outline, [largest_contour], -1, (0, 0, 255), 2)

    # --- Display results ---
    cv2.imshow("Original", img)
    cv2.imshow("Boundary mask (raw)", mask)
    cv2.imshow("Largest Contour Filled Mask", filled_mask)
    cv2.imshow("Largest Contour Outlined", outline)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return largest_contour, filled_mask


def segment_by_color(img, debug=False):
    """
    Segments the image based on color to separate red/yellow objects
    from a blue-gray background and shadows.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Convert to HSV (better separation of hue/lightness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges (tuned empirically)
    # Red can appear at both low and high hue ends (wrap-around)
    lower_red1 = np.array([0, 60, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 60])
    upper_red2 = np.array([180, 255, 255])

    # Cream-yellow tones
    lower_yellow = np.array([15, 30, 80])
    upper_yellow = np.array([45, 255, 255])

    # Combine masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    object_mask = cv2.bitwise_or(mask_red1, mask_red2)
    object_mask = cv2.bitwise_or(object_mask, mask_yellow)

    # Optional: remove noise, close gaps
    k = 15
    kernel = np.ones((k, k), np.uint8)
    mask_clean = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imshow("HSV Mask (object)", mask_clean)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_clean

def process_directory(folder_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    # Get all jpg, png, jpeg files
    image_paths = sorted(
        glob(os.path.join(folder_path, "*.jpg")) +
        glob(os.path.join(folder_path, "*.JPG")) +
        glob(os.path.join(folder_path, "*.png"))
    )

    batch = []  # store (img, mask, result, filename) for groups of 3
    counter = 1

    for img_path in image_paths:
        img = cv2.imread(img_path)

        segmented = segment_by_color(img, debug=True)
        contour_line, mask = find_contour(segmented)

        # Extract object
        result = cv2.bitwise_and(img, img, mask=mask)

        batch.append((img, mask, result, os.path.basename(img_path)))

        # When we have 3, show + save them
        if len(batch) == 3:
            show_and_save_triplet(batch, output_folder, counter)
            counter +=1
            batch = []

    # Handle leftover images (not divisible by 3)
    if batch:
        show_and_save_triplet(batch, output_folder)



def show_and_save_triplet(batch, output_folder, counter):
    # batch = [(img, mask, result, filename), ...]

    fig, axes = plt.subplots(len(batch), 3, figsize=(12, 4 * len(batch)))

    if len(batch) == 1:
        axes = [axes]  # make it iterable

    for i, (img, mask, result, filename) in enumerate(batch):
        # Convert BGR → RGB for plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        axes[i][0].imshow(img_rgb)
        axes[i][0].set_title(f"Original: {filename}")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(result_rgb)
        axes[i][2].set_title("Extracted Object")
        axes[i][2].axis("off")

    plt.tight_layout()

    # Save figure
    save_name = f"batch_{counter}.png"
    save_path = os.path.join(output_folder, save_name)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

    plt.show()
    plt.close()

# static counter for file naming
show_and_save_triplet.counter = 1

def process_single(filename):
    img = cv2.imread(filename)

    # img = denoise_image(img, sigma=5)
    # cv2.imshow("denoised", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    segmented = segment_by_color(img, debug=True)
    contour_line, mask = find_contour(segmented)

    cv2.imshow("new mask", mask)
    cv2.imwrite("mask.png",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Extract object
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Segmented object", result)
    cv2.imwrite("segmented.png",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_directory("./sample_img/", output_folder="./results/")
    # process_single("./sample_img/sample.jpg")



# To do
# segment by color
# watershed segmentation with r parameters
# make bulk processing app
# make image stitching
# 