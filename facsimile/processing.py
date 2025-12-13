import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from .src.segmentation import segment_by_color, find_contour, denoise_image, segment_hsv
from .src.binarization import binarize, all_binarize_algos
from .src.utils import read_image, save_image_cv

def process_directory(folder_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_paths = sorted(
        glob(os.path.join(folder_path, "*.jpg")) +
        glob(os.path.join(folder_path, "*.JPG")) +
        glob(os.path.join(folder_path, "*.png"))
    )

    batch = []  # store (img, mask, result, filename) for groups of 3
    counter = 1

    for img_path in image_paths:

        img = cv2.imread(img_path)
        # segmented = segment_by_color(img, debug=True)
        # contour_line, mask = find_contour(segmented)
        # result = cv2.bitwise_and(img, img, mask=mask)
        mask, result =  process_pipeline([img_path,""], DEFAULT_OPTIONS, output_folder,debug=True)

        batch.append((img, mask, result, os.path.basename(img_path)))

        # When we have 3, show + save them
        if len(batch) == 3:
            show_and_save_triplet(batch, output_folder, counter)
            counter +=1
            batch = []

    # Handle leftover images (not divisible by 3)
    if batch:
        show_and_save_triplet(batch, output_folder, counter)

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

# ------------------------------
# Full pipeline
# ------------------------------
def process_pipeline(uploaded_paths, options, output_folder, debug=False):
    img_path = (uploaded_paths[0])
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load image")
    # preprocessing step with bilateral filter

    # --- BINARIZATION ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_opts = options["binarize"]
    # bilateral filter preprocess
    bin_bilat = bin_opts["bilateral"]
    if bin_bilat["d"] > 0:
        gray = cv2.bilateralFilter(
            gray,
            d=bin_bilat["d"],
            sigmaColor=bin_bilat["sigma_color"],
            sigmaSpace=bin_bilat["sigma_space"],
        )
    # binarize
    binary_img = binarize(gray, func=bin_opts["method"])
    # postprocess 
    mk = bin_opts["median_k"] # median filter
    if mk and mk > 1:
        binary_img = cv2.medianBlur(binary_img, mk | 1)
    mok = bin_opts["morph_open_k"] # morph open
    if mok and mok > 0:
        kernel = np.ones((mok, mok), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    mck = bin_opts["morph_close_k"] # morph close
    if mck and mck > 0:
        kernel = np.ones((mck, mck), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # --- SEGMENTATION ---
    seg_opts = options["segment"]
    # bilateral filter preprocess
    seg_bilat = seg_opts["bilateral"]
    if seg_bilat["d"] > 0:
        gray = cv2.bilateralFilter(
            gray,
            d=seg_bilat["d"],
            sigmaColor=seg_bilat["sigma_color"],
            sigmaSpace=seg_bilat["sigma_space"],
        )
    # segment
    if seg_opts["method"] == "color":
        mask = segment_by_color(img)
    elif seg_opts["method"] == "kmeans":
        mask = segment_hsv(img)
    else:
        raise ValueError("invalid segmentation method")
    # postprocess
    mck = seg_opts["morph_close_k"] # morph close
    if mck and mck > 0:
        kernel = np.ones((mck, mck), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # find contour line of biggest mask part
    contour_line, mask = find_contour(mask)

    # --- COMPOSITE OUTPUT ---
    h, w = mask.shape
    output = np.full((h, w), 255, dtype=np.uint8)
    output[mask == 255] = binary_img[mask == 255]
    contour_pixels = contour_line > 0
    output[contour_pixels] = 0

    final = output.copy()

    # if contour_line is not None and options.get("overlay", True):
    #     edge_inv = cv2.bitwise_not(contour_line)
    #     final = cv2.bitwise_and(final, edge_inv)

    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(output_folder, out_name)
    if debug:
        cv2.imshow("new mask", mask)
        cv2.imwrite("mask.png",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        segmented = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("Segmented object", segmented)
        cv2.imwrite("segmented.png",segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Mask via Color select", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        save_image_cv(final, out_path)
    return out_name


DEFAULT_OPTIONS = {
    "binarize": {
        "method": "WOLF",
        "bilateral": {"d": 5, "sigma_color": 30, "sigma_space": 5},
        "median_k": 3,
        "morph_open_k": 3,
        "morph_close_k": 9,
    },
    "segment": {
        "method": "color",
        "bilateral": {"d": 5, "sigma_color": 50, "sigma_space": 7},
        "morph_close_k": 15,
    },
    "overlay": False,
}


if __name__ == "__main__":
    # process_directory("./data/", output_folder="./results/")
    # process_single("./data/original-1-7.JPG", method="color", debug=True)
    img = ["./data/original-1-2.JPG",""]
    process_pipeline(img, DEFAULT_OPTIONS, "./results",debug=True)


# To do
# make bulk processing app
# make ui more useful
# get customizer up
# color picker!