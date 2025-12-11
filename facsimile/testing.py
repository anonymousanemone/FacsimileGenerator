import os
import uuid
import numpy as np
import cv2
from .src.utils import read_image, save_image_cv

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

def segment_by_contour(img_gray, min_area=100):
    # _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((3,3), np.uint8)
    # th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask

def stitch_images(paths):
    return 'non existent'

# ------------------------------
# Full pipeline
# ------------------------------
def process_pipeline(uploaded_paths, options, output_folder):
    print(options)

    base_img = read_image(uploaded_paths[0])

    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    if options.get("denoise"):
        gray = denoise_image(gray, h=int(options.get("denoise_h", 10)))


    if options.get("segment"):
        min_area = int(options.get("segment_min_area", 100))
        mask = segment_by_contour(gray, min_area)
        gray = cv2.bitwise_and(gray, mask)

    if options.get("edge_detect"):
        low = int(options.get("canny_low", 50))
        high = int(options.get("canny_high", 150))
        edges = detect_edges(gray, low, high)
    else:
        edges = None

    method = options.get("binarize_method", "otsu")
    if method == "fixed":
        bw = binarize(gray, "fixed", int(options.get("binarize_thresh", 128)))
    elif method == "adaptive":
        win = int(options.get("adaptive_win", 11))
        if win % 2 == 0:
            win += 1
        bw = binarize(gray, "adaptive", adapt_win=win)
    else:
        bw = binarize(gray, "otsu")

    final = bw.copy()
    if edges is not None and options.get("edge_overlay", True):
        edge_inv = cv2.bitwise_not(edges)
        final = cv2.bitwise_and(final, edge_inv)

    if options.get("morph_clean"):
        ksize = int(options.get("morph_kernel", 3))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.ones((ksize, ksize), np.uint8)
        op = cv2.MORPH_OPEN if options.get("morph_op", "open") == "open" else cv2.MORPH_CLOSE
        final = cv2.morphologyEx(final, op, kernel, iterations=1)

    _, final = cv2.threshold(final, 127, 255, cv2.THRESH_BINARY)

    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(output_folder, out_name)
    save_image_cv(final, out_path)
    return out_name

"""
Tests for boundary and edge detection functions in processing.py.
Run with:
    python test_processing.py
"""

def test_boundary_detection_contours(img):
    """Test contour-based segmentation (boundary detection)."""
    print("[TEST] Boundary detection via contours")

    mask = segment_by_contour(img, min_area=200)

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
    k = 10
    kernel = np.ones((k, k), np.uint8)
    mask_clean = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imshow("HSV Mask (object)", mask_clean)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_clean

import cv2
import numpy as np

import cv2
import numpy as np

def segment_by_color_auto(img, 
                          n_colors=2, 
                          h_tol=15, s_tol=60, v_tol=60, 
                          blur_ksize=51,   # << VERY HIGH GAUSSIAN BLUR (must be odd)
                          debug=False):
    """
    Automatically segments the image by detecting dominant colors, with strong Gaussian blur.
    """

    # Ensure 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --- VERY STRONG BLUR TO SMOOTH COLORS ---
    # Large kernel (51x51) gives heavy smoothing; increase if needed.
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), sigmaX=0)

    # Convert blurred image to HSV
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Flatten for k-means
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # --- K-MEANS CLUSTERING ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        n_colors,
        None,
        criteria,
        attempts=5,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    # Build segmentation mask
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for c in centers:
        ch, cs, cv = int(c[0]), int(c[1]), int(c[2])

        lower = np.array([max(0, ch - h_tol), max(0, cs - s_tol), max(0, cv - v_tol)])
        upper = np.array([min(179, ch + h_tol), min(255, cs + s_tol), min(255, cv + v_tol)])

        mask = cv2.inRange(hsv, lower, upper)
        final_mask = cv2.bitwise_or(final_mask, mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    if debug:
        print("Detected dominant HSV centers:", centers)

        clustered = centers[labels.flatten()].reshape((h, w, 3))
        clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_HSV2BGR)
        cv2.imshow("Blurred", img_blur)
        cv2.imshow("Dominant Colors Visualization", clustered_bgr)
        cv2.imshow("Segmentation Mask", final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_mask


def segment_by_color_auto2(img, n_colors=2, h_tol=15, s_tol=60, v_tol=60, debug=False):
    """
    Automatically segments the image by detecting its two dominant colors
    using K-means in HSV space.

    n_colors  – number of dominant colors to detect (default=2)
    h_tol/s_tol/v_tol – tolerance around each dominant color center
    """

    # Ensure 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Flatten for k-means
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # --- K-MEANS CLUSTERING ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        n_colors,
        None,
        criteria,
        attempts=5,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)  # Convert cluster centers back to uint8 (HSV colors)

    # Create an empty mask
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # --- BUILD MASK FOR EACH DOMINANT COLOR ---
    for c in centers:
        ch, cs, cv = int(c[0]), int(c[1]), int(c[2])

        lower = np.array([
            max(0, ch - h_tol),
            max(0, cs - s_tol),
            max(0, cv - v_tol)
        ])

        upper = np.array([
            min(179, ch + h_tol),
            min(255, cs + s_tol),
            min(255, cv + v_tol)
        ])

        mask = cv2.inRange(hsv, lower, upper)
        final_mask = cv2.bitwise_or(final_mask, mask)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # Debug visualization
    if debug:
        print("Detected dominant HSV colors:", centers)

        clustered = centers[labels.flatten()].reshape((h, w, 3))
        clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_HSV2BGR)

        cv2.imshow("Dominant Colors Visualization", clustered_bgr)
        cv2.imshow("Segmentation Mask", final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_mask


import cv2
import os
import matplotlib.pyplot as plt
from glob import glob

def process_directory(folder_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    # Get all jpg, png, jpeg files
    image_paths = sorted(
        glob(os.path.join(folder_path, "*.jpg")) +
        glob(os.path.join(folder_path, "*.JPG")) +
        glob(os.path.join(folder_path, "*.png"))
    )

    batch = []  # store (img, mask, result, filename) for groups of 3

    for img_path in image_paths:
        img = cv2.imread(img_path)

        # Your segmentation function
        mask = segment_by_color(img, debug=True)

        # Extract object
        result = cv2.bitwise_and(img, img, mask=mask)

        batch.append((img, mask, result, os.path.basename(img_path)))

        # When we have 3, show + save them
        if len(batch) == 3:
            show_and_save_triplet(batch, output_folder)
            batch = []

    # Handle leftover images (not divisible by 3)
    if batch:
        show_and_save_triplet(batch, output_folder)


def show_and_save_triplet(batch, output_folder):
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
    save_name = f"batch_{show_and_save_triplet.counter}.png"
    save_path = os.path.join(output_folder, save_name)
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")

    plt.show()
    plt.close()

# static counter for file naming
show_and_save_triplet.counter = 1

import numpy as np
from skimage import filters, morphology, segmentation, measure, feature
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_gradient_magnitude, binary_fill_holes
import cv2


def watershed_background_segment(gray):
    h, w = gray.shape

    # --- 1. Convert to dataframe-like arrays ---
    y, x = np.indices(gray.shape)
    d = np.column_stack((x.ravel(), y.ravel(), gray.ravel()))

    # --- 2. Sample 10k pixels + fit background model ---
    idx = np.random.choice(len(d), size=min(10000, len(d)), replace=False)
    sample = d[idx]

    X = (sample[:,0] * sample[:,1]).reshape(-1,1)
    y_sample = sample[:,2]

    model = LinearRegression().fit(X, y_sample)

    # --- 3. Predict background and subtract ---
    X_full = (d[:,0] * d[:,1]).reshape(-1,1)
    bg_pred = model.predict(X_full).reshape(gray.shape)
    gray_c = gray - bg_pred

    # --- 4. Percent thresholds ---
    t10 = np.percentile(gray_c, 10)
    t90 = np.percentile(gray_c, 90)

    bg = gray_c < t10
    fg = gray_c > t90

    # --- 5. Combine seeds ---
    seed = np.zeros_like(gray, dtype=np.int32)
    seed[bg] = 1
    seed[fg] = 2

    # --- 6. Gradient magnitude (edge strength) ---
    edges = gaussian_gradient_magnitude(gray, sigma=1)

    # --- 7. Probabilities for watershed ---
    p = 1.0 / (1.0 + edges)

    # --- 8. Perform watershed ---
    ws = segmentation.watershed(p, seed) == 1

    # --- 9. Remove boundary flood (bucket fill) ---
    filled = morphology.flood_fill(ws, (0,0), False)
    ws = filled

    # --- 10. Fill holes ---
    ws = binary_fill_holes(ws)

    # --- 11. Remove tiny specks ---
    ws = morphology.remove_small_objects(ws, min_size=20)

    return ws.astype(np.uint8)


if __name__ == "__main__":

    process_directory("./data/", output_folder="./results/")
    # img = cv2.imread("./data/sample.jpg", cv2.IMREAD_GRAYSCALE) / 255.0
    # mask = watershed_background_segment(img)

    # cv2.imshow("mask", mask)
    # cv2.imwrite("mask.png", mask * 255)

    # img = cv2.imread("./data/sample.jpg")

    # img = denoise_image(img, sigma=6)
    # cv2.imshow("Edges", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    

    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # l2 = clahe.apply(l)
    # lab = cv2.merge((l2, a, b))
    # img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # cv2.imshow("Edges", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # mask = segment_by_color(img, debug=True)

    # # Extract object
    # result = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Segmented object", result)
    # cv2.imwrite("segmented.png",result)

    # test_boundary_detection_contours(mask)


# To do
# segment by color
# watershed segmentation with r parameters
# make bulk processing app
# make image stitching
# 