import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import img_stitch as solutions
import utils
import os

"""
Executable file for image stitch
"""


def challenge1e():
    # Read images
    imgs = []
    files = [
        utils.get_data_path("law-center.jpg"),
        utils.get_data_path("law-left.jpg"),
        utils.get_data_path("law-right.jpg"),
    ]
    for fname in files:
        img = utils.imread(fname, normalize=True)
        imgs.append(img)

    # Stitch images
    panorama = solutions.stitch_imgs(imgs)

    panorama = np.clip(panorama, 0, 1)
    utils.imshow(panorama)
    utils.imwrite(utils.get_result_path("1e.png"), panorama)


def stitchpanorama():
    # Panorama on your own images.
    panorama = solutions.build_your_own_panorama()

    panorama = np.clip(panorama, 0, 1)
    utils.imshow(panorama, flag=None) #comment this out at the end
    utils.imwrite(utils.get_result_path("facsimile_POST_IMG_STITCH.png"), panorama, flag=cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    stitchpanorama()