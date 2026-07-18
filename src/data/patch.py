import cv2
import os
import random
from pathlib import Path

def save_random_patches_opencv(
    image_path,
    output_dir,
    patch_size=(64, 64),
    num_patches=100,
    prefix="patch",
    seed=None
):
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    img_h, img_w = img.shape[:2]
    patch_w, patch_h = patch_size

    if patch_w > img_w or patch_h > img_h:
        raise ValueError(f"patch_size {patch_size} is larger than image size {(img_w, img_h)}.")

    max_x = img_w - patch_w
    max_y = img_h - patch_h

    for i in range(num_patches):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        patch = img[y:y+patch_h, x:x+patch_w]
        save_path = os.path.join(output_dir, f"{image_path.stem}_{prefix}_{i:04d}.png")
        cv2.imwrite(save_path, patch)

def main():
    data_split_path = Path('data/split')
    dirs = ['train', 'val', 'test']

    for d in dirs:
        for p in (data_split_path / Path(d)).iterdir():
            if not p.is_file():
                for t in p.iterdir():
                    save_random_patches_opencv(
                        image_path=t,
                        output_dir=t.parent.parent / Path('patches'),
                        patch_size=(256, 256),
                        num_patches=64,
                        seed=42,
                    )
                    print(f'Processed {t.stem}')

if __name__ == '__main__':
    main()