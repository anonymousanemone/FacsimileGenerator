import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_image(path, as_gray=False):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def save_image_cv(img, out_path):
    ext = os.path.splitext(out_path)[1]
    success, imbuf = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Failed to encode image {out_path}")
    imbuf.tofile(out_path)

def show_image_grid(images, titles=None, ncols=3, figsize=(12, 8), cmap='gray'):
    """
    Display a list of images in an n x m grid using matplotlib.

    Args:
        images (list[np.ndarray]): List of images to display.
        titles (list[str], optional): List of titles for each image.
        ncols (int, optional): Number of columns in the grid. Default is 3.
        figsize (tuple, optional): Figure size in inches. Default (12, 8).
        cmap (str, optional): Default color map for grayscale images.

    Example:
        show_image_grid(
            [img, denoised, binary, edges],
            ["Original", "Denoised", "Binary", "Edges"],
            ncols=2
        )
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n_images:
            img = images[i]
            # Handle grayscale vs color images
            if len(img.shape) == 2:  # grayscale
                ax.imshow(img, cmap=cmap)
            else:  # color, convert BGR → RGB for correct display
                import cv2
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
