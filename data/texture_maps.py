import yaml

from pathlib import Path
from PIL import Image

import cv2
import numpy as np

with open('config.yaml', 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

def load_grayscale_image(path):
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def save_grayscale_image(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8), mode='L').save(path)

def save_rgb_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path)

def extract_height_map(gray_img, blur_ksize=5, invert=False, normalize_output=True):
    height = gray_img.copy()

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        height = cv2.GaussianBlur(height, (blur_ksize, blur_ksize), 0)

    if invert:
        height = 1.0 - height

    if normalize_output:
        hmin, hmax = height.min(), height.max()
        if hmax > hmin:
            height = (height - hmin) / (hmax - hmin)

    return height

def extract_normal_map_rgb(height_map, strength=4.0, invert_y=False):
    h = height_map.astype(np.float32)

    dx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(h)

    normal = np.dstack((nx, ny, nz))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(norm, 1e-8)

    if invert_y:
        normal[:, :, 1] *= -1.0

    normal_rgb = (normal + 1.0) * 0.5
    normal_rgb = np.clip(normal_rgb, 0.0, 1.0)
    normal_rgb = (normal_rgb * 255.0).astype(np.uint8)

    return normal_rgb

def _is_valid_image(path):
    path = Path(path)

    if not path.exists():
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _maps_are_available_and_up_to_date(texture_path, height_path, normal_path):
    texture_path = Path(texture_path)
    height_path = Path(height_path)
    normal_path = Path(normal_path)

    if not _is_valid_image(height_path):
        return False

    if not _is_valid_image(normal_path):
        return False

    texture_mtime = texture_path.stat().st_mtime

    if height_path.stat().st_mtime < texture_mtime:
        return False

    if normal_path.stat().st_mtime < texture_mtime:
        return False

    return True

def process_texture(
    texture_path,
    output_dir="output_maps",
    save_texture_maps=False,
    blur_ksize=5,
    strength=4.0,
    invert=False,
    invert_y=False,
):
    if not isinstance(texture_path, Path):
        texture_path = Path(texture_path)

    # texture_path가 예를 들어
    #   data/split/train/patch/2_patch_0000.png
    # 라면 output_dir은
    #   data/split/train/output_maps
    # 가 됨
    output_dir = texture_path.parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    height_path = output_dir / f"{texture_path.stem}_height_map_gray.png"
    normal_path = output_dir / f"{texture_path.stem}_normal_map_rgb.png"

    maps_ready = _maps_are_available_and_up_to_date(
        texture_path=texture_path,
        height_path=height_path,
        normal_path=normal_path,
    )

    # save_texture_maps=True이면 강제 재생성
    # save_texture_maps=False여도 파일이 없거나 오래되었거나 깨져 있으면 생성
    if save_texture_maps or not maps_ready:
        gray_img = load_grayscale_image(texture_path)

        height_map = extract_height_map(
            gray_img,
            blur_ksize=blur_ksize,
            invert=invert,
            normalize_output=True,
        )

        normal_map_rgb = extract_normal_map_rgb(
            height_map,
            strength=strength,
            invert_y=invert_y,
        )

        save_grayscale_image(height_map, height_path)
        save_rgb_image(normal_map_rgb, normal_path)

    return str(height_path), str(normal_path)

'''
def process_texture(texture_path, output_dir="output_maps", save_texture_maps=False, blur_ksize=5, strength=4.0, invert=False, invert_y=False):
    if not isinstance(texture_path, Path):
        texture_path = Path(texture_path)
    output_dir = texture_path.parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    height_path = output_dir / f"{texture_path.stem}_height_map_gray.png"
    normal_path = output_dir / f"{texture_path.stem}_normal_map_rgb.png"

    if save_texture_maps:
        gray_img = load_grayscale_image(texture_path)

        height_map = extract_height_map(
            gray_img,
            blur_ksize=blur_ksize,
            invert=invert,
            normalize_output=True
        )

        normal_map_rgb = extract_normal_map_rgb(
            height_map,
            strength=strength,
            invert_y=invert_y
        )

        save_grayscale_image(height_map, height_path)
        save_rgb_image(normal_map_rgb, normal_path)

    return str(height_path), str(normal_path)
'''

if __name__ == "__main__":
    texture_path = "input_texture.png"

    height_path, normal_path = process_texture(
        texture_path,
        output_dir="output_maps",
        blur_ksize=5,
        strength=4.0,
        invert=False,
        invert_y=False
    )

    print("Saved height map:", height_path)
    print("Saved RGB normal map:", normal_path)