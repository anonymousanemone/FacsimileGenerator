
from skimage import data
from matplotlib import pyplot as plt
import skimage
import cv2
from PIL import Image, ImageFilter
import numpy as np

# https://pypi.org/project/doxapy/
import doxapy

algorithms = ['OTSU', 'BERNSEN', 'NIBLACK', 'SAUVOLA', 'WOLF', 'GATOS', 'NICK', 'SU', 'TRSINGH', 'BATAINEH', 'ISAUVOLA', 'WAN']

def read_image(file):
    im1 = Image.open(file).convert('L')
    im1 = im1.filter(ImageFilter.GaussianBlur(radius = 0.8)) 
    return np.array(im1)

def all_binarize_algos(gray_img):
    all_bins = []
    for i in range(len(algorithms)):
        binary_image = np.empty(gray_img.shape, gray_img.dtype)
        # Pick an algorithm from the DoxaPy library and convert the image to binary
        facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, algorithms[i]))
        facsimile.initialize(gray_img)
        facsimile.to_binary(binary_image)
        all_bins.append(binary_image)
    return all_bins

def  plot_all_binarize(img_path):
    # image = skimage.io.imread(img_path)
    # hsv_image = skimage.color.rgb2hsv(image)
    # val = hsv_image[:, :, 2]
    # grayscale_image = val

    grayscale_image = read_image(img_path)
    img_n = cv2.normalize(src=grayscale_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    images = all_binarize_algos(img_n)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(algorithms[i], fontsize=10)
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig("all_bin.png")
    plt.show()
    

def binarize(img_path, func="WOLF", debug=False):
    grayscale_image = read_image(img_path)

    binary_image = np.empty(grayscale_image.shape, grayscale_image.dtype)
    facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, func))
    facsimile.initialize(grayscale_image)
    facsimile.to_binary(binary_image)

    if debug:
        plt.imshow(binary_image, cmap="gray")
        skimage.io.imsave("./temp.png", binary_image)

    return binary_image
    

if __name__ == "__main__":
    plot_all_binarize("sample_img/sample.jpg")

