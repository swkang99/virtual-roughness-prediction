"""
8x8 grey-level co-occurrence matrix (8 grey levels: value // 32), flattened to 64 values.

mode="legacy"   : skimage graycomatrix, distance 1, angle 0, symmetric, normalised to sum 1
                  (used for the results so far).
mode="original" : Hassan et al. (Sci. Rep. 2023): "GLCM ... produces a matrix of 8 x 8 ... flattened to 1 x 64".
                  Details not given in the paper follow the authors' MATLAB code (glcmFeatureTrain100.m):
                  manual count of pairs (i, j) -> (i, j + 2) (horizontal, 2-pixel offset), non-symmetric,
                  normalised to sum 1, flattened column-major (MATLAB reshape). Verified against the
                  authors' stored features (r = 1.000).
                  The authors then min-max scale every GLCM feature to [0, 1] over the data set
                  (FeatureConcate.m). That step needs training statistics and is applied per training
                  fold in test/cv_experiment.py (GLCM_SLICE_ORIGINAL), never here.
"""
import numpy as np
from skimage.feature import graycomatrix
from skimage import color

GLCM_OFFSET_ORIGINAL = 2


def _to_gray(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)   # (1, H, W) -> (H, W)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = color.rgb2gray(img)   # (H, W, 3) -> (H, W)
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image shape for GLCM: {img.shape}")
    return img


def gray_level_co_occurrence_matrix(img, mode="legacy"):
    img = _to_gray(img)

    # Quantization in the range of 0–7 (8 levels)
    img_q = np.clip((img // 32).astype(np.int64), 0, 7)

    if mode == "original":
        s = GLCM_OFFSET_ORIGINAL
        g = np.bincount((img_q[:, :-s] * 8 + img_q[:, s:]).ravel(), minlength=64).reshape(8, 8).astype(np.float64)
        return g / g.sum()

    glcm = graycomatrix(
        img_q.astype(np.uint8),
        distances=[1],
        angles=[0],
        levels=8,             # Level 8 → 8×8 Matrix
        symmetric=True,
        normed=True
    )
    return glcm[:, :, 0, 0].astype(np.float64)


def flatten_glcm(glcm_2d, mode="legacy"):
    return glcm_2d.flatten(order="F" if mode == "original" else "C")
