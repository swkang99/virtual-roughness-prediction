"""
Normalize Tier-based texture/height/normal map triplets to a common square resolution,
then extract fixed-size random patches for training.

Background: Hassan et al. (2023, Scientific Reports) textures (Tier B here) were photographed
under per-texture, uncontrolled conditions, so native resolution varies substantially
(roughly 1140-2725 px across the 90 Tier B textures) and a minority of photos are not square.
Because the true physical scale (mm/px) of each photo cannot be recovered after the fact, we
adopt a documented, deterministic normalization instead of trying to estimate it: each
texture/height/normal triplet is center-cropped to a square (using the shorter side, so no
anisotropic stretching) and resized to a common TARGET_SIZE, before sampling patches.

Usage (run once per Tier, from repo root):

    python -m src.data.prepare_tier_patches --tier "Tier B"
    python -m src.data.prepare_tier_patches --tier "Tier A'"
    python -m src.data.prepare_tier_patches --tier "Tier A"

Or all three tiers in one call:

    python -m src.data.prepare_tier_patches --all

Output layout (written under data/trustworthy/<Tier>/):

    Normalized/Texture Image/<stem>.<ext>        (square, TARGET_SIZE x TARGET_SIZE)
    Normalized/Height Map/<stem>_height.png      (single-channel, TARGET_SIZE x TARGET_SIZE)
    Normalized/Normal Map/<stem>_normal.png      (RGB, TARGET_SIZE x TARGET_SIZE)
    Patches/Texture/<stem>_patch_XXXX.png                         (PATCH x PATCH, RGB)
    Patches/Height/<stem>_patch_XXXX_height_map_gray.png          (PATCH x PATCH, grey)
    Patches/Normal/<stem>_patch_XXXX_normal_map_rgb.png           (PATCH x PATCH, RGB)

`Normalized/` is a deterministic function of the original (publicly available, CC BY 4.0 for
the Hassan-sourced Tier B/A' textures) data and is intended to accompany the public data release.
`Patches/` is for local training only and should stay out of version control (see .gitignore):
with every normalized image now the same TARGET_SIZE x TARGET_SIZE square, `patch_coords()` below
produces the exact same 64 (x, y) positions for every single texture (fixed seed + fixed image
size), so the full patch set is trivially and exactly reproducible from `Normalized/` alone with
this script. It is also the heavier artifact on disk: 64 overlapping 256x256 patches cover more
total pixels (64 * 256 * 256) than one 1600x1600 source image.
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_SIZE = 1600
PATCH = 256
N_PATCHES = 64
SEED = 42

TIERS = ["Tier B", "Tier A'", "Tier A"]
TEX_EXT = {".jpg", ".jpeg", ".png"}

try:  # Pillow >= 9.1 moved these under Image.Resampling
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BICUBIC = Image.BICUBIC

# Tier A's "Texture Image" files keep their original APT capture names (r<roughness>_<name>.jpg),
# which bear no string relationship to the A1..A8 ids assigned to the Height Map / Normal Map
# files in this repo (see data/trustworthy/Tier A/Tier A.csv). This mapping was derived once by
# visually matching each height map's texture pattern against the real photos (see project notes)
# and is fixed, so it is spelled out here rather than guessed at by string matching.
TIER_A_TEXTURE_OVERRIDE = {
    "A1_Silk": "r3_silk",
    "A2_Leather": "r5_leather",
    "A3_Felt": "r20_felt",
    "A4_Hemp": "r40_hemp",
    "A5_Bath Towel": "r40_bath(cotton)_towel",
    "A6_Sandpaper (120-grit)": "r75_sand_paper_#120",
    "A7_Scrubber": "r80_scrubber",
    "A8_Exfoliating Towel": "r87_exfoliating_towel",
}


def patch_coords(size=TARGET_SIZE, patch=PATCH, n=N_PATCHES, seed=SEED):
    """Deterministic patch top-left coordinates. Identical for every texture, since every
    normalized image is the same TARGET_SIZE x TARGET_SIZE square."""
    random.seed(seed)
    return [(random.randint(0, size - patch), random.randint(0, size - patch)) for _ in range(n)]


def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_height(path):
    """Collapse Photoshop's RGB-triplicated (R=G=B) height maps and true single-channel
    grayscale height maps (see project notes on the Tier A/A' vs Tier B export difference)
    to the same single-channel representation."""
    img = Image.open(path)
    a = np.asarray(img)
    if a.ndim == 3:
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError(f"height map is not grey (R, G, B differ): {path}")
        return Image.fromarray(a[..., 0], mode="L")
    return img.convert("L")


def _stem_without_suffix(path, suffix):
    name = path.stem
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _find_texture(stem, tex_by_stem, override):
    if stem in tex_by_stem:
        return tex_by_stem[stem]
    # e.g. height stem "Q1_Sandpaper" vs texture stem "Q1_Sandpaper (150)" - the texture image
    # alone carries an extra grit/spec annotation that the height/normal maps dropped.
    prefix_hits = [k for k in tex_by_stem if k.startswith(stem + " ") or k.startswith(stem + "_")]
    if len(prefix_hits) == 1:
        return tex_by_stem[prefix_hits[0]]
    if stem in override and override[stem] in tex_by_stem:
        return tex_by_stem[override[stem]]
    return None


def discover_triplets(tier_dir):
    """Match Texture Image / Height Map / Normal Map files by their shared '<id>_<name>' stem."""
    tex_dir = tier_dir / "Texture Image"
    height_dir = tier_dir / "Height Map"
    normal_dir = tier_dir / "Normal Map"

    height_by_stem = {_stem_without_suffix(p, "_height"): p for p in height_dir.glob("*.png")}
    normal_by_stem = {_stem_without_suffix(p, "_normal"): p for p in normal_dir.glob("*.png")}
    tex_by_stem = {p.stem: p for p in tex_dir.iterdir() if p.suffix.lower() in TEX_EXT}

    override = TIER_A_TEXTURE_OVERRIDE if tier_dir.name == "Tier A" else {}

    missing_normal = sorted(set(height_by_stem) ^ set(normal_by_stem))
    if missing_normal:
        raise RuntimeError(f"{tier_dir.name}: height/normal map sets do not match 1:1: {missing_normal}")

    triplets, unresolved = [], []
    for stem, height_path in sorted(height_by_stem.items()):
        tex_path = _find_texture(stem, tex_by_stem, override)
        if tex_path is None:
            unresolved.append(stem)
            continue
        triplets.append((stem, tex_path, height_path, normal_by_stem[stem]))

    if unresolved:
        raise RuntimeError(f"{tier_dir.name}: no texture image found for {unresolved}")

    return triplets


def normalize_and_patch(tier_dir, write=True):
    triplets = discover_triplets(tier_dir)
    norm_root = tier_dir / "Normalized"
    patch_root = tier_dir / "Patches"
    for sub in ("Texture Image", "Height Map", "Normal Map"):
        (norm_root / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("Texture", "Height", "Normal"):
        (patch_root / sub).mkdir(parents=True, exist_ok=True)

    coords = patch_coords()
    n_patches_written = 0

    for stem, tex_path, height_path, normal_path in triplets:
        tex_img = center_crop_square(Image.open(tex_path).convert("RGB")).resize(
            (TARGET_SIZE, TARGET_SIZE), RESAMPLE_LANCZOS
        )
        height_img = center_crop_square(load_height(height_path)).resize(
            (TARGET_SIZE, TARGET_SIZE), RESAMPLE_BICUBIC
        )
        normal_img = center_crop_square(Image.open(normal_path).convert("RGB")).resize(
            (TARGET_SIZE, TARGET_SIZE), RESAMPLE_BICUBIC
        )

        if write:
            tex_ext = re.sub(r"^\.", "", tex_path.suffix.lower()) or "jpg"
            tex_img.save(norm_root / "Texture Image" / f"{stem}.{tex_ext}")
            height_img.save(norm_root / "Height Map" / f"{stem}_height.png")
            normal_img.save(norm_root / "Normal Map" / f"{stem}_normal.png")

        tex_arr = np.asarray(tex_img)
        height_arr = np.asarray(height_img)
        normal_arr = np.asarray(normal_img)

        for i, (x, y) in enumerate(coords):
            if write:
                Image.fromarray(tex_arr[y:y + PATCH, x:x + PATCH], mode="RGB").save(
                    patch_root / "Texture" / f"{stem}_patch_{i:04d}.png"
                )
                Image.fromarray(height_arr[y:y + PATCH, x:x + PATCH], mode="L").save(
                    patch_root / "Height" / f"{stem}_patch_{i:04d}_height_map_gray.png"
                )
                Image.fromarray(normal_arr[y:y + PATCH, x:x + PATCH], mode="RGB").save(
                    patch_root / "Normal" / f"{stem}_patch_{i:04d}_normal_map_rgb.png"
                )
            n_patches_written += 1

    print(
        f"[{tier_dir.name}] {len(triplets)} textures normalized to {TARGET_SIZE}x{TARGET_SIZE}, "
        f"{n_patches_written} patches {'written' if write else 'would be written'} "
        f"-> {norm_root}, {patch_root}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/trustworthy")
    ap.add_argument("--tier", choices=TIERS, help="process a single tier, e.g. \"Tier B\"")
    ap.add_argument("--all", action="store_true", help="process all three tiers")
    args = ap.parse_args()

    if not args.tier and not args.all:
        ap.error('pass --tier "Tier B" (or "Tier A'"'"'" / "Tier A") or --all')

    for tier in (TIERS if args.all else [args.tier]):
        normalize_and_patch(Path(args.data_root) / tier, write=True)


if __name__ == "__main__":
    main()
