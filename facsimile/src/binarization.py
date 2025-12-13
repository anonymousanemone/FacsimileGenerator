
from skimage import data
from matplotlib import pyplot as plt
import skimage
import cv2
import numpy as np

# https://pypi.org/project/doxapy/
import doxapy

algorithms = ['OTSU', 'BERNSEN', 'NIBLACK', 'SAUVOLA', 'WOLF', 'GATOS', 'NICK', 'SU', 'TRSINGH', 'BATAINEH', 'ISAUVOLA', 'WAN']

def all_binarize_algos(gray_img: np.ndarray):
    all_bins = []
    for algo in range(len(algorithms)):
        binary_image = np.empty(gray_img.shape, gray_img.dtype)
        # Pick an algorithm from the DoxaPy library and convert the image to binary
        facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, algo))
        facsimile.initialize(gray_img)
        facsimile.to_binary(binary_image)
        all_bins.append(binary_image)
    return all_bins

def  plot_all_binarize(gray_img: np.ndarray):
    images = all_binarize_algos(gray_img)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(algorithms[i], fontsize=10)
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig("all_bin.png")
    plt.show()
    

def binarize(gray_img: np.ndarray, func="WOLF", debug=False):
    binary_image = np.empty(gray_img.shape, gray_img.dtype)
    facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, func))
    facsimile.initialize(gray_img)
    facsimile.to_binary(binary_image)

    if debug:
        plt.imshow(binary_image, cmap="gray")
        skimage.io.imsave("./temp.png", binary_image)

    return binary_image
    

if __name__ == "__main__":
    gray = cv2.imread("sample_img/sample.jpg", cv2.IMREAD_GRAYSCALE)
    plot_all_binarize(gray)

