import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode

# CONFIG_FILE = "color_config.json"
CONFIG_FILE = "nope.json"


DEFAULT_CONFIG = {
    "color_ranges": [
        {
            "name": "lower_red1",
            "lower": [0, 60, 60],
            "upper": [10, 255, 255]
        },
        {
            "name": "upper_red1",
            "lower": [170, 60, 60],
            "upper": [180, 255, 255]
        },
        {
            "name": "yellow",
            "lower": [15, 30, 80],
            "upper": [45, 255, 255]
        }
    ]
}


def load_color_config():
    """Load color config or fall back to defaults."""
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_color_config(config):
    """Save updated color configuration to JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def denoise_image(img, sigma=6.0):
    k = int(np.ceil(2 * np.pi * sigma)) // 2 * 2 + 1
    denoised = cv2.GaussianBlur(img, (k, k), sigma)
    return denoised

def segment_by_color(img, morph_close=15, debug=False):
    """
    Segments the image based on color definitions stored in JSON.
    """
    # Load config (user or default)
    config = load_color_config()
    color_ranges = config["color_ranges"]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Apply all color bounds from config
    for cr in color_ranges:
        lower = np.array(cr["lower"])
        upper = np.array(cr["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # close holes
    # kernel = np.ones((morph_close, morph_close), np.uint8)
    # mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imshow("Mask via Color select", combined_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined_mask

def find_contour(mask, debug=False):
    # find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"  Found {len(contours)} contour(s)")
    assert len(contours) > 0, "No contours found — boundary detection failed."

    # get largest contour and fill it in
    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    contour_line = np.zeros_like(mask)
    cv2.drawContours(contour_line, [largest_contour], -1, 255, thickness=2)
    
    if debug:
        # draw contour on img
        if len(mask.shape) == 2:
            outline = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            outline = mask.copy()
        cv2.drawContours(outline, [largest_contour], -1, (0, 0, 255), 2)

        cv2.imshow("Original", mask)
        cv2.imshow("Boundary mask (raw)", mask)
        cv2.imshow("Largest Contour Filled Mask", filled_mask)
        cv2.imshow("Largest Contour Outlined", outline)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contour_line, filled_mask

def kmeans_segment(img, features, k=2, r=10):
    h, w = img.shape[:2]

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_.reshape(h, w)

    # Find majority label in rxr center
    cx, cy = w // 2, h // 2
    roi = labels[cy - r:cy + r, cx - r:cx + r]

    m = mode(roi.flatten(), keepdims=False).mode
    majority_label = int(m)

    binary = np.uint8(labels == majority_label) * 255
    return labels, binary, majority_label


def segment_hsv(img, select_hsv="HSV", morph_close=15, output_file=None, debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Pre-filtering ---
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_blur = cv2.medianBlur(img_blur, 5)

    h, w = img_blur.shape[:2]
    rgb = img_blur.reshape(-1, 3)

    # HSV feature set
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    hsv = hsv_img.reshape(-1, 3)
    
    if select_hsv == "H":
        hsv_selected = hsv[:, [0]]
    elif select_hsv == "S":
        hsv_selected = hsv[:, [1]]
    elif select_hsv == "V":
        hsv_selected = hsv[:, [2]]
    else:
        hsv_selected = hsv

    # RGB + HSV kmeans
    rgb_hsv = np.column_stack((rgb, hsv_selected))
    labels, mask, maj_labels = kmeans_segment(
        img_blur, rgb_hsv, k=2
    )
    # close holes
    # kernel = np.ones((morph_close, morph_close), np.uint8)
    # mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imshow("Mask via Kmeans with seHSVlect", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plot
        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        # axes[0].set_title("Original")
        # axes[0].imshow(img_blur)
        # axes[0].axis("off")

        # axes[1].set_title("KMeans (RGB + %s)" % select_hsv)
        # axes[1].imshow(labels, cmap="nipy_spectral")
        # axes[1].axis("off")


        # axes[2].set_title("KMeans (RGB + %s) binary" % select_hsv)
        # axes[2].imshow(mask, cmap="nipy_spectral")
        # axes[2].axis("off")

        # plt.tight_layout()

        # save if file name
        # if output_file:
        #     plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.show()
        # plt.close()
    return mask

def process_single(img, method="color", debug=False):
    # preprocess step
    img = denoise_image(img, sigma=5)
    # cv2.imshow("denoised", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if method=="color":
        segmented = segment_by_color(img, debug=debug)
    elif method=="kmeans":
        segmented = segment_hsv(img, debug=debug)
    else:
        print("invalid method")
        return
    contour_line, mask = find_contour(segmented)

    if debug:
        cv2.imshow("new mask", mask)
        cv2.imwrite("mask.png",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Extract object
    result = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow("Segmented object", result)
        cv2.imwrite("segmented.png",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("../data/sample.jpg")
    process_single(img, method="color", debug=True)