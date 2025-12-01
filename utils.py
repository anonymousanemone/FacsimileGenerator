import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def print_banner(s: str = ""):
    """
    Print a banner to stdout.

    Args:
        s (str): Banner heading.
    """
    if s:
        print(s)
    try:
        banner_width = int(os.popen("stty size", "r").read().split()[-1])
    except IndexError:
        banner_width = 30
    print("=" * banner_width)


def get_data_path(filename: str) -> str:
    """
    Return the path to a data file.

    Args:
        filename (str): Name of the file.

    Returns:
        str: The absolute path to the file.
    """
    return str((DATA_DIR / filename).resolve())


def get_result_path(filename: str) -> str:
    """
    Return the path to a data file.

    Args:
        filename (str): Name of the file.

    Returns:
        str: The absolute path to the file.
    """
    return str((RESULTS_DIR / filename).resolve())


def imread(
    path: str, flag: int = cv2.IMREAD_COLOR, rgb: bool = False, normalize: bool = False
) -> np.ndarray:
    """
    Read an image from a file.

    Args:
        path (str): Path to the image.
        flag (int, optional): Image read flag passed to cv2.imread. Defaults to cv2.IMREAD_COLOR.
        rgb (bool, optional): Whether to read the image as BGR (rgb=False) or RGB (rgb=True).
            Defaults to False.
        normalize (bool, optional): Normalize the image to the range [0, 1]. Defaults to False.

    Raises:
        FileNotFoundError: If the filepath could not be found.

    Returns:
        np.ndarray: The image as a multidimensional array.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(str(path), flag)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255
    return img


def imread_alpha(path: str, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an image from a file.
    Use this function when the image contains an alpha channel. That channel
    is returned separately.

    Args:
        path (str): Path to the image.
        normalize (bool, optional): Normalize the image to the range [0, 1]. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the image and alpha channel.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if normalize:
        img = img.astype(np.float32) / 255

    alpha = img[:, :, -1]
    img = img[:, :, :-1]

    return img, alpha


def imshow(img: np.ndarray, title: str = None, flag: int = cv2.COLOR_BGR2RGB):
    """
    Display an image in a windowed viewer.

    Args:
        img (np.ndarray): Input image.
        title (str, optional): Title of the window. Defaults to None.
        flag (int, optional): cv2.cvtColor flag. Defaults to cv2.COLOR_BGR2RGB.
    """
    plt.figure()
    if flag is not None:
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        img = cv2.cvtColor(img, flag)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def imwrite(path: str, img: np.ndarray, flag: Optional[int] = None):
    """
    Write an image to a file.

    Args:
        path (str): Path to save the image to.
        img (np.ndarray): The input image.
        flag (Optional[int], optional): cv2.cvtColor flag. Defaults to None.
    """
    assert type(img) == np.ndarray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    cv2.imwrite(str(path), img)


def sift_matches(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtain point correspondence using SIFT features.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: src_points (Nx2) of points in img1,
            dst_points (Nx2) correspondences in img2.
    """
    if img1.dtype == np.float64 or img1.dtype == np.float32:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype == np.float64 or img2.dtype == np.float32:
        img2 = (img2 * 255).astype(np.uint8)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    src_pts = np.asarray([kp1[good[i][0].queryIdx].pt for i in range(len(good))])
    dest_pts = np.asarray([kp2[good[i][0].trainIdx].pt for i in range(len(good))])

    return src_pts, dest_pts


def show_correspondences(
    src_img: np.ndarray,
    dest_img: np.ndarray,
    src_pts: np.ndarray,
    dest_pts: np.ndarray,
    title: Optional[str] = None,
    show_every_n: int = 1,
) -> Figure:
    """
    Visualize correspondences between two images by plotting both images
    side-by-side and drawing lines between each point correspondence.

    Since the correspondences may be dense, show_every_n controls how many
    point correspondences to show.

    Args:
        src_img (np.ndarray): Source image.
        dest_img (np.ndarray): Destination image.
        src_pts (np.ndarray): Point correspondences in the source image (Nx2).
        dest_pts (np.ndarray): Point correspondences in the destination image (Nx2).
        title (Optional[str], optional): Window title. Defaults to None.
        show_every_n (int, optional): How many points to show. Defaults to 1.

    Returns:
        Figure: Matplotlib figure plotting the correspondence points.
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    N = src_pts.shape[0]

    fig, ax = plt.subplots()
    plt.axis("off")

    padding = 80
    ax.imshow(
        np.hstack(
            (src_img, np.full((src_img.shape[0], padding, 3), 1, np.float32), dest_img)
        )
    )
    t = src_img.shape[1] + padding
    for i in range(0, N, show_every_n):
        # Draw line
        xs = src_pts[i, :]
        xd = dest_pts[i, :]
        ax.plot([xs[0], xd[0] + t], [xs[1], xd[1]], "r-", linewidth=0.75)

    if title is not None:
        plt.title(title)

    return fig


class EventHandler:
    def __init__(self, ax):
        self.ax = ax

        self.keypress = self.keypress_factory(self.ax)

    def keypress_factory(self, ax):
        def event_exit_manager(event):
            if event.key in ["enter"]:
                plt.close()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("key_press_event", event_exit_manager)

        return event_exit_manager


class ZoomPanEventHandler(EventHandler):
    def __init__(self, ax):
        super().__init__(ax)
        self.press = None
        self.xpress = None
        self.ypress = None

        self.original_xlim = None
        self.original_ylim = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

        self.keypress = self.keypress_factory(self.ax)
        self.zoom = self.zoom_factory(self.ax, base_scale=1.25)
        self.pan = self.pan_factory(self.ax)

    def keypress_factory(self, ax):
        def keypress_callback(event):
            if event.key in ["enter"]:
                plt.close()
            elif event.key == "r":
                if self.original_xlim is not None and self.original_ylim is not None:
                    ax.set_xlim(self.original_xlim)
                    ax.set_ylim(self.original_ylim)
                    ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("key_press_event", keypress_callback)

        return keypress_callback

    def zoom_factory(self, ax, base_scale=2.0):
        def zoom(event):
            if self.original_xlim is None:
                self.original_xlim = ax.get_xlim()
                self.original_ylim = ax.get_ylim()
            if event.inaxes != ax:
                return

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location
            if event.button == "up":
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == "down":
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest
        fig.canvas.mpl_connect("scroll_event", zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax:
                return
            if event.button == 3:
                self.cur_xlim = ax.get_xlim()
                self.cur_ylim = ax.get_ylim()
                self.press = self.x0, self.y0, event.xdata, event.ydata
                self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None:
                return
            if event.inaxes != ax:
                return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("button_press_event", onPress)
        fig.canvas.mpl_connect("button_release_event", onRelease)
        fig.canvas.mpl_connect("motion_notify_event", onMotion)

        # return the function
        return onMotion


class SelectPointsEventHandler(ZoomPanEventHandler):
    def __init__(self, ax, points: List[Tuple[int, int]] = [], max_points: int = 100):
        super().__init__(ax)
        self.points = list(points)
        self.max_points = max_points
        self.count = 0
        self.keypress = self.keypress_factory(self.ax)
        self.point_select = self.point_select_factory(self.ax)

        self.plotted_points = []

        if len(self.points) > 0:
            self.plotted_points.append(ax.scatter(*zip(*self.points), c="r"))
            fig = self.ax.get_figure()
            fig.canvas.draw()

    def keypress_factory(self, ax):
        def keypress_callback(event):
            if event.key in ["enter"]:
                plt.close()
            elif event.key == "c":
                self.points.clear()
                for _ in range(len(self.plotted_points)):
                    plotted_point = self.plotted_points.pop(0)
                    plotted_point.remove()
                    ax.figure.canvas.draw()
            elif event.key == "r":
                if self.original_xlim is not None and self.original_ylim is not None:
                    ax.set_xlim(self.original_xlim)
                    ax.set_ylim(self.original_ylim)
                    ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("key_press_event", keypress_callback)

        return keypress_callback

    def point_select_factory(self, ax):
        def point_select_callback(eclick):
            "eclick is the press event"
            if eclick.inaxes != ax:
                return
            if eclick.button == 1:
                if self.count == 0 and len(self.points) > 0:
                    self.points.clear()
                    for _ in range(len(self.plotted_points)):
                        plotted_point = self.plotted_points.pop(0)
                        plotted_point.remove()

                x1, y1 = eclick.xdata, eclick.ydata
                self.plotted_points.append(ax.scatter(x1, y1, c="r"))

                fig = ax.get_figure()
                fig.canvas.draw()

                self.points.append([x1, y1])
                self.count += 1
                if len(self.points) >= self.max_points:
                    plt.close()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect("button_press_event", point_select_callback)

        return point_select_callback


class SelectPointsViewer:
    def __init__(
        self,
        img: np.ndarray,
        max_points: int,
        existing_points: List[Tuple[int, int]] = [],
        suptitle: str = "",
        window_title: str = "SelectPointsViewer",
        **plt_kwargs,
    ):
        self.img = img
        self.fig, self.ax = plt.subplots(**plt_kwargs)
        if suptitle:
            self.fig.suptitle(suptitle)
        self.fig.canvas.manager.set_window_title(window_title)
        self.ax.imshow(self.img, cmap="gray")
        self.eh = SelectPointsEventHandler(self.ax, existing_points, max_points)

    def show(self):
        plt.show()

    def get_points(self) -> np.ndarray:
        return np.array(self.eh.points).astype(int)


def click_pts(
    img: np.ndarray,
    N: int,
    title: Optional[str] = None,
    prev_points: List[Tuple[int, int]] = [],
) -> np.ndarray:
    """
    Helper function to select points on an image by clicking.

    Args:
        img (np.ndarray): Image to display.
        N (int): Number of points to click.
        title (Optional[str], optional): Title of the window. Defaults to None.

    Returns:
        np.ndarray: (Nx2) array of (x, y) coordinates of clicked points.
    """
    viewer = SelectPointsViewer(
        img,
        max_points=N,
        existing_points=prev_points,
        suptitle=title,
        # subplot_kw={
        #     "xticks": [],
        #     "yticks": [],
        #     "frame_on": False,
        # },
    )
    controls = (
        "left click: select point"
        + "    scroll: zoom in/out"
        + "    right click + drag: pan"
        + "\nr: reset zoom/pan"
        + "    c: clear point selection"
        + "    enter: submit selected points"
    )
    viewer.fig.text(0.5, 0.895, controls, ha="center", fontsize=8)
    viewer.show()
    return viewer.get_points()
