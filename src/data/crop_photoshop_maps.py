"""
Prepare the Photoshop-generated height / normal maps for training (run once, repo root):

    python -m src.data.crop_photoshop_maps [--src data/shuffled_withmaps] [--data_root data] [--remove_old_maps]

1. Re-number   data/shuffled_withmaps/<n>.jpg is matched to the original texture data/<split>/texture_image/<id>.jpg
               by identical file bytes (SHA-256); the map pair <n>.png is copied unchanged to
               data/texture_maps/{height,normal}/<id>.png (original numbering), mapping -> data/texture_maps/mapping.csv
2. Verify      every saved texture patch is re-cropped from its source image with the coordinates of
               src/data/patch.py (random.seed(42) per image) and must be pixel-identical; maps must have the
               image resolution. Nothing is written unless every check passes.
3. Crop        the maps at the same 256 x 256 positions -> data/<split>/ps_maps/<id>_patch_XXXX_height_map_gray.png
               (8-bit grey) and ..._normal_map_rgb.png (8-bit RGB). Pixel values are copied, not re-normalised,
               so training sees the same map data as Unity.
4. (optional)  --remove_old_maps deletes the old Python-generated patch maps data/<split>/output_maps/.
"""
import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PATCH = 256
N_PATCHES = 64
SEED = 42
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def patch_coords(img_w, img_h, n=N_PATCHES, size=PATCH, seed=SEED):
    """Same sequence as src/data/patch.py: random.seed(seed); x = randint(0, W-size); y = randint(0, H-size)."""
    random.seed(seed)
    out = []
    for _ in range(n):
        x = random.randint(0, img_w - size)
        y = random.randint(0, img_h - size)
        out.append((x, y))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def images_by_id(folder):
    return {int(p.stem): p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT and p.stem.isdigit()}


def load_height(path):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:                                   # Photoshop saves grey height maps as RGB with R = G = B
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError(f"height map is not grey (R, G, B differ): {path}")
        a = a[..., 0]
    if a.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit height map, got {a.dtype}: {path}")
    return a


def load_normal(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    if a.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit normal map, got {a.dtype}: {path}")
    return a


# ----------------------------------------------------------------------------- 1. re-number
def renumber(src, data_root, out_root):
    originals = {}
    for split in SPLITS:
        for tid, p in images_by_id(Path(data_root) / split / "texture_image").items():
            originals[sha256(p)] = (tid, split, p)
    shuffled = images_by_id(src)
    mapping = []
    for sid, p in sorted(shuffled.items()):
        hit = originals.get(sha256(p))
        if hit is None:
            raise RuntimeError(f"{p} matches no original texture image")
        mapping.append((sid, hit[0], hit[1]))
    ids = [m[1] for m in mapping]
    if len(set(ids)) != len(ids) or len(ids) != len(originals):
        raise RuntimeError(f"mapping is not one-to-one ({len(set(ids))} unique of {len(originals)} originals)")

    for kind, sub in (("height", "HeightMap"), ("normal", "NormalMap")):
        (out_root / kind).mkdir(parents=True, exist_ok=True)
        for sid, tid, _ in mapping:
            shutil.copy2(Path(src) / sub / f"{sid}.png", out_root / kind / f"{tid}.png")
    with open(out_root / "mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_id", "split", "shuffled_id"])
        for sid, tid, split in sorted(mapping, key=lambda m: m[1]):
            w.writerow([tid, split, sid])
    print(f"[renumber] {len(mapping)} map pairs -> {out_root}/height|normal/<original id>.png, mapping.csv")


# ----------------------------------------------------------------------------- 2./3. verify + crop
def crop(data_root, maps_root, out_name, write):
    n = 0
    for split in SPLITS:
        pdir = Path(data_root) / split / "patches"
        odir = Path(data_root) / split / out_name
        originals = images_by_id(Path(data_root) / split / "texture_image")
        tex_ids = sorted({int(p.name.split("_")[0]) for p in pdir.glob("*_patch_*.png")})
        for tid in tex_ids:
            img = cv2.imread(str(originals[tid]))                   # BGR, as in patch.py
            h, w = img.shape[:2]
            hmap = load_height(maps_root / "height" / f"{tid}.png")
            nmap = load_normal(maps_root / "normal" / f"{tid}.png")
            if hmap.shape != (h, w) or nmap.shape[:2] != (h, w):
                raise ValueError(f"texture {tid}: image {w}x{h}, height {hmap.shape[::-1]}, normal {nmap.shape[1::-1]}")
            for i, (x, y) in enumerate(patch_coords(w, h)):
                stem = f"{tid}_patch_{i:04d}"
                saved = cv2.imread(str(pdir / f"{stem}.png"))
                if saved is None or not np.array_equal(saved, img[y:y + PATCH, x:x + PATCH]):
                    raise RuntimeError(f"patch position check failed for {split}/{stem} - nothing written")
                if write:
                    odir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(hmap[y:y + PATCH, x:x + PATCH], mode="L").save(odir / f"{stem}_height_map_gray.png")
                    Image.fromarray(nmap[y:y + PATCH, x:x + PATCH], mode="RGB").save(odir / f"{stem}_normal_map_rgb.png")
                n += 1
        print(f"[{'crop' if write else 'verify'}] {split}: {len(tex_ids)} textures")
    print(f"[{'crop' if write else 'verify'}] {n} patches " + (f"written to data/<split>/{out_name}/" if write else "OK"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/shuffled_withmaps", help="shuffled images + HeightMap/ + NormalMap/")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--maps_root", default="data/texture_maps", help="re-numbered full-size maps")
    ap.add_argument("--out_name", default="ps_maps", help="per-split folder of cropped map patches")
    ap.add_argument("--skip_renumber", action="store_true", help="maps_root already filled")
    ap.add_argument("--remove_old_maps", action="store_true", help="delete data/<split>/output_maps afterwards")
    a = ap.parse_args()

    maps_root = Path(a.maps_root)
    if not a.skip_renumber:
        renumber(a.src, a.data_root, maps_root)
    crop(a.data_root, maps_root, a.out_name, write=False)          # every check first
    crop(a.data_root, maps_root, a.out_name, write=True)
    if a.remove_old_maps:
        for split in SPLITS:
            old = Path(a.data_root) / split / "output_maps"
            if old.exists():
                shutil.rmtree(old)
                print(f"[cleanup] removed {old}")


if __name__ == "__main__":
    main()
