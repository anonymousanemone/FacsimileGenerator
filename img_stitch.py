from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import utils
from scipy.ndimage import distance_transform_edt
import os

# Tunable parameters
# RANSAC parameters 
CHALLENGE_1C_RANSAC_N = 500  # Number of iterations
CHALLENGE_1C_RANSAC_EPS = 5  # Maximum reprojection error
""" 
Image stitching for facsimiles of scrolls, pottery and ceramics that 
need to have multiple facsimile section stitched together 
into one longer image
"""


#HW 4 functions
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix relating the given points.
    Hint: use np.linalg.eig to compute the eigenvalues of a matrix.

    Args:
        src_pts (np.ndarray): Nx2 matrix of source points
        dest_pts (np.ndarray): Nx2 matrix of destination points

    Returns:
        np.ndarray: 3x3 homography matrix
    """
    """assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2
    N = src_pts.shape[0]"""

    #Ah = 0, solving for h such that ||h||^2 = 1
    #in A_transpose*Ah = lambda*h
    #1) take A_transpose*A, find eigenvalues and eigenvectors
    A_rows = []
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        x_, y_ = dest_pts[i]

        A_rows.append([-x, -y, -1, 0,0,0, x* x_, y*x_, x_])
        A_rows.append([0, 0, 0, -x,-y,-1, x* y_, y*y_, y_])

    A_rows = np.array(A_rows)
    A_transpose = A_rows.T #A_transpose

    #print(f"A_transpose: {A_transpose}")

    Ata =A_transpose @ A_rows #Ata = A_transpose*A

    Ata_eivalue, Ata_eivector = np.linalg.eig(Ata)


    #2) h is the eigenvector corresponding to the smallest eigenvalue in A_transpose*A

    min_eigenval_idx = np.argmin(Ata_eivalue)
    #print(f"min_eigenval_idx: {min_eigenval_idx}")
    h = Ata_eivector[:, min_eigenval_idx]
    #print(f"min value: {h}")

    #3) rearrange h into a 3x3 matrix
    h_matrix = h.reshape((3,3))
    h_matrix = h_matrix/h_matrix[2,2]

    #print(f"h_matrix: {h_matrix}")

    return h_matrix


def apply_homography(H: np.ndarray, test_pts: np.ndarray) -> np.ndarray:
    """
    Apply the homography to the given test points

    Args:
        H (np.ndarray): 3x3 homography matrix
        test_pts (np.ndarray): Nx2 test points

    Returns:
        np.ndarray: Nx2 points after applying the homography
    """
    # 1) convert to homogenous coordinates, p=[x,y,1] ^T

    #print(f"test_pts: {test_pts}")
    homogenous_test_pts = np.column_stack([test_pts, np.ones(len(test_pts))])
    homogenous_test_pts_T = homogenous_test_pts.T
    #print(f"homogenous_test_pts_T: {homogenous_test_pts.T}")

    #print(f"H: {H}")
    
    #2) Apply homography, p'=H⋅p
    transformed_test_pts = H@homogenous_test_pts_T

    #3) convert back to cartesian coords
    transformed_test_pts = transformed_test_pts.T #need to convert matrix back to og untransposed format

    cartesian_transformed = transformed_test_pts[:,:2]/transformed_test_pts[:,2:3]
    #print(f"cartesian: {cartesian_transformed}")
    #print(f"test_pts: {test_pts.shape[0]}, cartesian_pts: {cartesian_transformed.shape[0]}")

    return cartesian_transformed
    #assert test_pts.shape[1] == 2


def backward_warp_img(
    src_img: np.ndarray, H_inv: np.ndarray, dest_canvas_width_height: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a homography to the image using backward warping.

    Use cv2.remap to interpolate the warped points.
    The function call should follow this form:
        img_warp = cv2.remap(img, map_x.astype(np.float32),
            map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=np.nan).
    This casts map_x and map_y to 32-bit floats, chooses linear interpolation,
    and sets pixels outside the original image to NaN (not-a-number).

    Also, since we are working with color images, you should process each
    color channel separately.

    Args:
        src_img (np.ndarray): Nx2 source points
        H_inv (np.ndarray): 3x3 inverse of the src -> dest homography
        dest_canvas_width_height (Tuple[int, int]): size of the destination image

    Returns:
        Tuple[np.ndarray, np.ndarray]: final image, binary mask where the
            destination image is filled in
    """
    #iterate through output image and apply inverse transformation
    print(f"H_inv: {H_inv}")
    dest_width, dest_height = dest_canvas_width_height

    x_out, y_out = np.meshgrid(np.arange(dest_width), np.arange(dest_height))
    coords_out = np.column_stack([x_out.ravel(), y_out.ravel()])
    coords_in = apply_homography(H_inv, coords_out)

    map_x = coords_in[:, 0].reshape(dest_height, dest_width)
    map_y = coords_in[:, 1].reshape(dest_height, dest_width)
    
    img_warp = cv2.remap(src_img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=np.nan)
    binary_mask = ~np.isnan(img_warp).any(axis=2)
    return img_warp, binary_mask



def warp_img_onto(
    src_img: np.ndarray, dest_img: np.ndarray, src_pts: np.ndarray, dest_pts: np.ndarray
) -> np.ndarray:
    """
    Warp the source image on the destination image.
    Return the resulting image.
   
    Args:
        src_img (np.ndarray): source image
        dest_img (np.ndarray): destination image
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points

    Returns:
        np.ndarray: resulting image with the source image warped on the
        destination image
    """

    #Step 1: estimate the homography mapping src_pts to dest_pts
    estimated_homography = compute_homography(src_pts, dest_pts)
    inv_homography = np.linalg.inv(estimated_homography) #I think I can use this method?
    dest_width,dest_height = dest_img.shape[:2]


    #Step 2: warp src_img onto dest_img using backward_warp_img(..)
    warped_img, img_mask = backward_warp_img(src_img, inv_homography, (dest_height,dest_width))

    print(f"width: {dest_width}")
    print(f"height: {dest_height}")

    print(f"dest_img shape: {dest_img.shape}")
    print(f"warped_img shape: {warped_img.shape}") 
    print(f"img_mask shape: {img_mask.shape}")

    #Step 3: overlay the warped src_img onto dest_img
    result = dest_img.copy()
    result[img_mask] = warped_img[img_mask]

    #Step 4: return the resulting image
    return result


def run_RANSAC(
    src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, ransac_eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run RANSAC on the given point correspondences to compute a homography
    from source to destination points. When sampling new points at each RANSAC
    iteration, remember to sample _without_ replacement.

    Args:
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points
        ransac_n (int): number of RANSAC iterations
        ransac_eps (float): maximum 2D reprojection error for inliers

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the inlier indices and the
        estimated homography matrix.
    """
    
    s = 4 #change
    M_inliers = []
    best_inlier_count = 0
    best_homography = []
    inlier_idx = []
    #4. Repeat Steps 1-3 𝑁 times
    for j in range(ransac_n):
        #1. Randomly choose 𝑠 samples. Typically 𝑠 is the minimum samples to fit a model.
        samples_idx = np.random.choice(len(src_pts), s, replace = False)
        sample_src = src_pts[samples_idx]
        sample_dest = dest_pts[samples_idx]

        #2. Fit the model to the randomly chosen samples.
        sample_H = compute_homography(sample_src, sample_dest)

        #3. Count the number 𝑀 of data points (inliers) that fit the model within a measure of error 𝜀.
        current_inlier_idx = []
        #euclidean distance
        for i in range(len(src_pts)):
            src_reshape = src_pts[i].reshape(1,2) #so format fits apply_homography()
            predicted_dest = apply_homography(sample_H, src_reshape)
            predicted_dest = predicted_dest[0]
            #print(f"sample h: {sample_H}, src pts: {src_pts[i]}")
            error = np.linalg.norm(predicted_dest-dest_pts[i])
            if error < ransac_eps:
                current_inlier_idx.append(i)

        if len(current_inlier_idx) > best_inlier_count:
            best_homography = sample_H.copy()
            best_inlier_count = len(current_inlier_idx)
            inlier_idx = current_inlier_idx.copy()
        
    #5. Choose the model that has the largest number 𝑀 of inliers
    best_src = src_pts[inlier_idx]
    best_dest = dest_pts[inlier_idx]
    src_to_dest_H = compute_homography(best_src, best_dest)

    return inlier_idx, src_to_dest_H

    #assert src_pts.shape[0] == dest_pts.shape[0]
    #assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2



def blend_image_pair(
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
    blending_mode: str,
) -> np.ndarray:
    """
    Blend two images together using the "overlay" or "blend" mode.
    Args:
        img1 (np.ndarray): First image
        mask1 (np.ndarray): Mask where the first image is non-zero
        img2 (np.ndarray): Second image
        mask2 (np.ndarray): Mask where the second image is non-zero
        blending_mode (str): "overlay" or "blend"

    Returns:
        np.ndarray: blended image.
    """
    assert blending_mode in ["overlay", "blend"]
    if blending_mode=="overlay":
        #copy img2 over img1 wherever the mask2 applies
        mask2_bool = mask2.astype(bool)
        img1[mask2_bool] = img2[mask2_bool]
        return img1
    if blending_mode=="blend":
        #weight blending
        mask1_bool = mask1.astype(bool)
        mask2_bool = mask2.astype(bool)
        
        w_1 = distance_transform_edt(mask1_bool).astype(float)
        w_2 = distance_transform_edt(mask2_bool).astype(float)
        w_1 = w_1 + 1e-10 # epsilon
        w_2 = w_2 + 1e-10
        w_1 = w_1[:, :, np.newaxis]
        w_2 = w_2[:, :, np.newaxis]
        
        # I_blend = (w_1*I_1 + w_2*I_2) / (w_1 + w_2)
        I_blend = (w_1 * img1 + w_2 * img2) / (w_1 + w_2)

        return I_blend



def stitch_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Stitch a list of images together into an image mosaic.
    imgs: list of images to be stitched together. You may assume the order
    the images appear in the list is the order in which they should be stitched
    together.

    Args:
        imgs (List[np.ndarray]): list of images to be stitched together. You may
        assume the order the images appear in the list is the order in which
        they should be stitched together.

    Returns:
        np.ndarray: the final, stitched image
    """
    
    if len(imgs) == 0:
        return np.array([])
    if len(imgs) == 1:
        return imgs[0]
    
    # 1) start with the first image as the base mosaic
    mosaic = imgs[0].copy()
    mosaic_mask = np.ones(mosaic.shape[:2], dtype=bool)
    
    for i in range(1, len(imgs)):
        next_img = imgs[i]
        
        # 2) find SIFT correspondences between next image and current mosaic
        src_pts, dest_pts = utils.sift_matches(next_img, mosaic)
        
        print(f"Stitching image {i} .... Found {len(src_pts)} SIFT matches")
        
        # 3) use RANSAC to find stronger homography
        inlier_idx, H = run_RANSAC(src_pts, dest_pts, CHALLENGE_1C_RANSAC_N, CHALLENGE_1C_RANSAC_EPS)
        
        #print(f"  RANSAC found {len(inlier_idx)} inliers out of {len(src_pts)} matches")
        
        # find corners of the next image after warping
        h_next, w_next = next_img.shape[:2]
        corners_next = np.array([[0, 0], [w_next-1, 0], [w_next-1, h_next-1], [0, h_next-1]], dtype=float)
        warped_corners = apply_homography(H, corners_next)
        
        # find bounding box for the warped image and current mosaic
        h_mosaic, w_mosaic = mosaic.shape[:2]
        mosaic_corners = np.array([[0, 0], [w_mosaic-1, 0], [w_mosaic-1, h_mosaic-1], [0, h_mosaic-1]], dtype=float)
        
        all_corners = np.vstack([mosaic_corners, warped_corners])
        x_min = int(np.floor(np.min(all_corners[:, 0])))
        x_max = int(np.ceil(np.max(all_corners[:, 0]))) + 1
        y_min = int(np.floor(np.min(all_corners[:, 1])))
        y_max = int(np.ceil(np.max(all_corners[:, 1]))) + 1
        
        # new size
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        
        #shift everything to positive coordinates
        T = np.eye(3)
        T[0, 2] = -x_min #translation matrix
        T[1, 2] = -y_min
        H_translated = T @ H #apply to homography
        
        #warp next img to new canvas
        warped_next, warped_next_mask = backward_warp_img(next_img, 
                                                           np.linalg.inv(H_translated),
                                                           (canvas_width, canvas_height))
        
        # Replace NaN values with 0
        warped_next = np.nan_to_num(warped_next, nan=0.0)
        
        # Place the current mosaic on the new canvas with translation
        canvas = np.zeros((canvas_height, canvas_width, mosaic.shape[2]), dtype=mosaic.dtype)
        canvas_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
        mosaic_x_offset = int(-x_min)
        mosaic_y_offset = int(-y_min)
        
        canvas[mosaic_y_offset:mosaic_y_offset+h_mosaic, 
               mosaic_x_offset:mosaic_x_offset+w_mosaic] = mosaic
        canvas_mask[mosaic_y_offset:mosaic_y_offset+h_mosaic,
                    mosaic_x_offset:mosaic_x_offset+w_mosaic] = mosaic_mask
        
        # Blend 
        mosaic = blend_image_pair(canvas, canvas_mask, warped_next, warped_next_mask, "blend")
        mosaic_mask = canvas_mask | warped_next_mask
    
    return mosaic


def build_your_own_panorama() -> np.ndarray:
    """
    regular panorama inputs in image_stitching/results/image_stitch_INPUTS.
    """
    input_path = utils.get_result_path("image_stitch_INPUTS")  # Do not change

    # Load images
    #file_names = ["2.png", "1.png", "3.png"]  
    #convert it all to pngs?
    
    #rename file names, for img in file names, name it i.png
    imgs = []
    for f_name in os.listdir(input_path):
        img_path = str((Path(input_path) / f_name).resolve())
        img = utils.imread(img_path, flag=None, rgb=True, normalize=True)
        
        #downsampling
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_w, new_h = 1280, int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        imgs.append(img)
    #final stitch
    panorama = stitch_imgs(imgs)
    
    return panorama