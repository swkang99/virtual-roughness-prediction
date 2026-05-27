import numpy as np
from skimage.feature import graycomatrix
from skimage import color, img_as_ubyte


def gray_level_co_occurrence_matrix(img):

    # grayscale 변환
    if img.ndim == 3:
        img = color.rgb2gray(img)
        img = img_as_ubyte(img)

    # img = resize(img, (1568, 1568), preserve_range=True, anti_aliasing=True)

    # 0–7 범위로 양자화 (8단계)
    img_q = (img // 32).astype(np.uint8)  # 256 → 8단계 (0~7)
    img_q = np.clip(img_q, 0, 7)  # range 0~7

    # GLCM 계산 (8×8 출력)
    glcm = graycomatrix(
        img_q,
        distances=[1],
        angles=[0],           # 원하면 [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels=8,             # 8단계 → 8×8 매트릭스
        symmetric=True,
        normed=True
    )

    # shape = (8, 8, 1, 1) 이므로, 2D로 줄이기
    glcm_2d = glcm[:, :, 0, 0]

    # print("GLCM shape:", glcm_2d.shape)  # (8, 8)
    # print("GLCM matrix:\n", glcm_2d)

    return glcm_2d