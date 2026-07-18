import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix
from skimage import io, color, img_as_ubyte

# 이미지 로드
img = io.imread("image.jpg")

if img.ndim == 3:
    img = color.rgb2gray(img)
    img = img_as_ubyte(img)

# 0–7 범위로 양자화 (8단계)
img_q = (img // 32).astype(np.uint8)
img_q = np.clip(img_q, 0, 7)

# 8×8 GLCM 계산
glcm = graycomatrix(
    img_q,
    distances=[1],
    angles=[0],
    levels=8,
    symmetric=True,
    normed=True
)

# 2D로 축 꺾기
glcm_2d = glcm[:, :, 0, 0]  # shape: (8, 8)

# GLCM 이미지로 시각화
plt.figure(figsize=(6, 5))
plt.imshow(glcm_2d, cmap="hot", interpolation="nearest")
plt.colorbar(label="Co-occurrence frequency")
plt.title("GLCM 8x8 Matrix (as Image)")
plt.xlabel("Gray level (i)")
plt.ylabel("Gray level (j)")
plt.xticks(range(8))
plt.yticks(range(8))
plt.show()