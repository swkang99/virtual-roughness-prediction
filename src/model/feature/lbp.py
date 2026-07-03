import numpy as np
import cv2

def build_uniform_lbp_mapping():
    mapping = {}
    uniform_patterns = []
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        circular = bits + [bits[0]]
        transitions = sum(circular[i] != circular[i+1] for i in range(8))
        if transitions <= 2:
            uniform_patterns.append(code)

    uniform_patterns = sorted(uniform_patterns)
    for idx, code in enumerate(uniform_patterns):
        mapping[code] = idx

    non_uniform_bin = 58
    return mapping, non_uniform_bin

UNIFORM_MAP, NON_UNIFORM_BIN = build_uniform_lbp_mapping()

def lbp_code_8neighbors(cell):
    h, w = cell.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = cell[y, x]
            code = 0
            code |= (cell[y-1, x-1] >= c) << 7
            code |= (cell[y-1, x  ] >= c) << 6
            code |= (cell[y-1, x+1] >= c) << 5
            code |= (cell[y  , x+1] >= c) << 4
            code |= (cell[y+1, x+1] >= c) << 3
            code |= (cell[y+1, x  ] >= c) << 2
            code |= (cell[y+1, x-1] >= c) << 1
            code |= (cell[y  , x-1] >= c) << 0
            lbp[y - 1, x - 1] = code

    return lbp

def lbp_hist_59(cell):
    lbp_raw = lbp_code_8neighbors(cell)
    hist = np.zeros(59, dtype=np.float32)

    for code in lbp_raw.ravel():
        bin_idx = UNIFORM_MAP.get(int(code), NON_UNIFORM_BIN)
        hist[bin_idx] += 1

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist, lbp_raw

def extract_lbp_feature(image, grid=(7, 7)):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the input image size rather than assuming 224x224.
    rows, cols = grid
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    features = []
    lbp_maps = []

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

    feature_vector = np.concatenate(features, axis=0)
    return feature_vector, lbp_maps