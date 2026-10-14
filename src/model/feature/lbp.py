"""
LBP features: 7x7 grid of cells, 59-bin histogram per cell -> 2891 values.

mode="legacy"   : the implementation used for the results so far. 8 integer neighbours (radius 1),
                  neighbour >= centre, uniform-pattern mapping, remainder pixels go to the last cell,
                  histogram / sum. The vectorised code reproduces the old per-pixel Python loop
                  bit-exactly (verified on real 256x256 patches and synthetic arrays with ties).

mode="original" : Hassan et al. (Sci. Rep. 2023) - "LBP ... thresholding a circular neighbourhood",
                  59 values per cell. The authors' stored features were produced with MATLAB
                  extractLBPFeatures (uniform, upright, radius 1, 8 neighbours, bilinear interpolation)
                  with 'Normalization','None' followed by division by the cell sum. Reproduced with
                  skimage 'nri_uniform' (circular, bilinear) + MATLAB bin order (UNIFORM_TO_MATLAB,
                  derived by matching the authors' CSV: r = 0.98) + histogram / sum.
                  Cells: floor(H/7) x floor(W/7) pixels, remainder dropped (MATLAB CellSize behaviour);
                  the 1-pixel image border is handled by symmetric padding.
"""
import numpy as np
import cv2


def build_uniform_lbp_mapping():
    mapping = {}
    uniform_patterns = []
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        circular = bits + [bits[0]]
        transitions = sum(circular[i] != circular[i + 1] for i in range(8))
        if transitions <= 2:
            uniform_patterns.append(code)

    uniform_patterns = sorted(uniform_patterns)
    for idx, code in enumerate(uniform_patterns):
        mapping[code] = idx

    non_uniform_bin = 58
    return mapping, non_uniform_bin


UNIFORM_MAP, NON_UNIFORM_BIN = build_uniform_lbp_mapping()

# 256-entry lookup table: LBP code -> histogram bin (58 uniform bins + 1 non-uniform bin)
UNIFORM_LUT = np.full(256, NON_UNIFORM_BIN, dtype=np.int64)
for _code, _idx in UNIFORM_MAP.items():
    UNIFORM_LUT[_code] = _idx

# (dy, dx, bit) in the same order as the original loop:
# top-left=7, top=6, top-right=5, right=4, bottom-right=3, bottom=2, bottom-left=1, left=0
_NEIGHBOURS = [(-1, -1, 7), (-1, 0, 6), (-1, 1, 5), (0, 1, 4), (1, 1, 3), (1, 0, 2), (1, -1, 1), (0, -1, 0)]

# skimage 'nri_uniform' bin -> MATLAB extractLBPFeatures bin (fitted on the authors' stored LBP features)
UNIFORM_TO_MATLAB = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2, 9, 16, 15, 14, 13, 12, 11, 10, 21, 24, 23, 22, 17, 20, 19,
                              18, 25, 32, 31, 30, 29, 28, 27, 26, 33, 36, 39, 38, 37, 40, 35, 34, 41, 44, 43, 46, 45,
                              48, 47, 42, 49, 56, 51, 54, 53, 52, 55, 50, 57, 58])


# ----------------------------------------------------------------------------- legacy
def _shift(cell, dy, dx):
    """View of `cell` shifted by (dy, dx), aligned with the interior cell[1:-1, 1:-1]."""
    h, w = cell.shape
    return cell[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]


def lbp_code_8neighbors(cell):
    """8-neighbour LBP codes of the interior pixels, shape (h-2, w-2), uint8 (legacy)."""
    cell = np.asarray(cell)
    centre = _shift(cell, 0, 0)
    code = np.zeros(centre.shape, dtype=np.uint8)
    for dy, dx, bit in _NEIGHBOURS:
        code |= ((_shift(cell, dy, dx) >= centre).astype(np.uint8) << bit)
    return code


def lbp_hist_59(cell):
    lbp_raw = lbp_code_8neighbors(cell)
    hist = np.bincount(UNIFORM_LUT[lbp_raw.ravel()], minlength=59).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist, lbp_raw


def _extract_legacy(image, grid):
    rows, cols = grid
    h, w = image.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    features, lbp_maps = [], []
    for r in range(rows):
        row_maps = []
        for c in range(cols):
            # For the last row/col include any remainder pixels
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < cols - 1 else w
            cell = image[y0:y1, x0:x1]

            # Ensure each cell is at least 3x3 for 8-neighbor LBP computation.
            ch, cw = cell.shape[:2]
            if ch < 3 or cw < 3:
                top = max(0, (3 - ch) // 2)
                bottom = max(0, 3 - ch - top)
                left = max(0, (3 - cw) // 2)
                right = max(0, 3 - cw - left)
                cell = cv2.copyMakeBorder(cell, top, bottom, left, right, cv2.BORDER_REPLICATE)

            hist, lbp_raw = lbp_hist_59(cell)
            features.append(hist)
            row_maps.append(lbp_raw)
        lbp_maps.append(row_maps)
    return np.concatenate(features, axis=0), lbp_maps


# ----------------------------------------------------------------------------- original (Hassan et al.)
def _extract_original(image, grid):
    from skimage.feature import local_binary_pattern
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    padded = np.pad(img, 1, mode="symmetric")
    codes = local_binary_pattern(padded, P=8, R=1, method="nri_uniform").astype(np.int64)[1:-1, 1:-1]
    codes = UNIFORM_TO_MATLAB[codes]

    rows, cols = grid
    h, w = img.shape
    ch, cw = h // rows, w // cols
    features, lbp_maps = [], []
    for r in range(rows):
        row_maps = []
        for c in range(cols):
            cell = codes[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            hist = np.bincount(cell.ravel(), minlength=59).astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            features.append(hist)
            row_maps.append(cell)
        lbp_maps.append(row_maps)
    return np.concatenate(features, axis=0), lbp_maps


def extract_lbp_feature(image, grid=(7, 7), mode="legacy"):
    """image: 2-D grey array (3-channel input is converted with cv2 BGR->GRAY, legacy behaviour)."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "original":
        return _extract_original(image, grid)
    return _extract_legacy(image, grid)
