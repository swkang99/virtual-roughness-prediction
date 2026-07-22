import numpy as np
from skimage.feature import graycomatrix
from skimage import color


def gray_level_co_occurrence_matrix(img):

    img = np.asarray(img)

    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)   # (1, H, W) -> (H, W)

    elif img.ndim == 3 and img.shape[-1] == 3:
        img = color.rgb2gray(img)   # (H, W, 3) -> (H, W)

    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape for GLCM: {img.shape}")

    # Quantization in the range of 0–7 (8 levels)
    img_q = (img // 32).astype(np.uint8)  # 256 → Level 8 (0–7)
    img_q = np.clip(img_q, 0, 7)  # range 0~7

    # GLCM Calculation (8×8 Output)
    glcm = graycomatrix(
        img_q,
        distances=[1],
        angles=[0],           # If desired, [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels=8,             # Level 8 → 8×8 Matrix
        symmetric=True,
        normed=True
    )

    # Since shape = (8, 8, 1, 1), reduce it to 2D
    glcm_2d = glcm[:, :, 0, 0]

    # print("GLCM shape:", glcm_2d.shape)  # (8, 8)
    # print("GLCM matrix:\n", glcm_2d)

    return glcm_2d