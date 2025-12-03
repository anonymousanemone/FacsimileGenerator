import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from src.segmentation import segment_by_color, find_contour, denoise_image
from src.binarization import binarize, all_binarize_algos

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
        mask, result = process_single(img_path)

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

def process_single(filename, bin_algo="WOLF", morph_close=15, debug=False):
    binary_img = binarize(filename)
    # pos process step here
    img = cv2.imread(filename)

    # img = denoise_image(img, sigma=5)
    # cv2.imshow("denoised", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = segment_by_color(img, debug=debug)
    contour_line, mask = find_contour(mask)

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
    
    result = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    if debug:
        cv2.imshow("final", result)
        cv2.imwrite("result.png",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask, result


if __name__ == "__main__":
    # process_directory("./sample_img/", output_folder="./results/")
    process_single("./sample_img/sample.jpg", debug=True)


# To do
# segment by color - done ish
# make bulk processing app
# make image stitching
# binarizing!