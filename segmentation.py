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


def segment_by_color(img, debug=False):
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

    # Clean mask
    k = 15
    kernel = np.ones((k, k), np.uint8)
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imshow("HSV Mask (object)", mask_clean)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_clean
