import cv2
import numpy as np
import json
import os

CONFIG_FILE = "color_config.json"

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

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Apply all color bounds from config
    for cr in color_ranges:
        lower = np.array(cr["lower"])
        upper = np.array(cr["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # close holes
    kernel = np.ones((morph_close, morph_close), np.uint8)
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imshow("HSV Mask (object)", mask_clean)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_clean

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

    return largest_contour, filled_mask


def process_single(filename, debug=False):
    img = cv2.imread(filename)

    # img = denoise_image(img, sigma=5)
    # cv2.imshow("denoised", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    segmented = segment_by_color(img, debug=True)
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
    process_single("./sample_img/sample.jpg")