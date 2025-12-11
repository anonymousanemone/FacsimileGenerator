import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from src.segmentation import segment_by_color, find_contour, denoise_image
from src.binarization import binarize, all_binarize_algos
from src.utils import read_image, save_image_cv

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
    binary_img = binarize(filename, func=bin_algo)
    # pos process step here
    img = cv2.imread(filename)

    # img = denoise_image(img, sigma=5)
    # cv2.imshow("denoised", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = segment_by_color(img, debug=debug)
    contour_line, mask = find_contour(mask)
    # cv2.imshow("contour_line", contour_line)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # do contour outline on white instead of black bg
    h, w = mask.shape
    output = np.full((h, w), 255, dtype=np.uint8)
    output[mask == 255] = binary_img[mask == 255]
    contour_pixels = contour_line > 0
    output[contour_pixels] = 0

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

        cv2.imshow("final", output)
        cv2.imwrite("result.png",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask, output

# ------------------------------
# Full pipeline
# ------------------------------
def process_pipeline(uploaded_paths, options, output_folder):
    img_path = (uploaded_paths[0])

    # denoise gaus option
    binary_img = binarize(img_path, func=options.get("binarize_method"))
    # median filter
    # morph open
    # morph close

    img = cv2.imread(img_path)
    denoised = denoise_image(img, h=int(options.get("denoise_gaus", 10)))
    mask = segment_by_color(denoised, morph_close=options.get("morph_close"))
    contour_line, mask = find_contour(mask)

    h, w = mask.shape
    output = np.full((h, w), 255, dtype=np.uint8)
    output[mask == 255] = binary_img[mask == 255]
    contour_pixels = contour_line > 0
    output[contour_pixels] = 0

    final = output.copy()
    # if contour_line is not None and options.get("overlay", True):
    #     edge_inv = cv2.bitwise_not(edges)
    #     final = cv2.bitwise_and(final, edge_inv)

    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(output_folder, out_name)
    save_image_cv(final, out_path)
    return out_name

if __name__ == "__main__":
    process_directory("./data/", output_folder="./results/")
    # process_single("./data/original-1-7.JPG", debug=True)


# To do
# make bulk processing app
# make ui more useful
# get customizer up
# color picker!