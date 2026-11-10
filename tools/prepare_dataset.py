"""Crop each photo pair to 16:9 and resize it to the three benchmark sizes.

Authors: Quách Châu Hạo Kiệt, Tôn Thất Minh Đăng
Drafted with an AI assistant (Claude) and reviewed by the team.

Input layout (one folder per site, two photos named left.* and right.*):
    raw/indoor/left.jpg   raw/indoor/right.jpg
    raw/outdoor/left.jpg  ...
Output layout:
    dataset/indoor/small/left.jpg   (1280x720)
    dataset/indoor/medium/left.jpg  (1920x1080)
    dataset/indoor/large/left.jpg   (3840x2160)

Usage: python tools/prepare_dataset.py raw dataset
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

SIZES = {"small": (1280, 720), "medium": (1920, 1080), "large": (3840, 2160)}


def find_photo(site: Path, name: str) -> Path:
    matches = [p for p in site.iterdir() if p.stem.lower() == name and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not matches:
        sys.exit(f"{site}: missing {name}.jpg / {name}.png")
    return matches[0]


def crop_16_9(img: Image.Image) -> Image.Image:
    """Centre crop to 16:9."""
    w, h = img.size
    if w * 9 > h * 16:          # too wide: trim the sides
        cw, ch = h * 16 // 9, h
    else:                       # too tall: trim top and bottom
        cw, ch = w, w * 9 // 16
    # symmetric centering, trim equally from top, bot, left, right.
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    sites = sorted(p for p in args.raw.iterdir() if p.is_dir())
    if not sites:
        sys.exit(f"no site folders in {args.raw}")
    for site in sites:
        for name in ("left", "right"):
            # exif_transpose applies the phone's rotation flag so portrait/landscape is correct
            img = ImageOps.exif_transpose(Image.open(find_photo(site, name))).convert("RGB")
            img = crop_16_9(img)
            if img.size[0] < SIZES["large"][0]:
                print(f"  note: {site.name}/{name} is {img.size[0]}x{img.size[1]} after cropping; 'large' is upscaled")
            for size_name, size in SIZES.items():
                dst = args.out / site.name / size_name / f"{name}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Lanczos filtering to avoid aliasing (jagged edges, may corrupt Harris response)
                img.resize(size, Image.Resampling.LANCZOS).save(dst, quality=95)
        print(f"{site.name}: done")


if __name__ == "__main__":
    main()
