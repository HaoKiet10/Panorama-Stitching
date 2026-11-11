"""Generate reference homography and stitched panorama using OpenCV.

Authors: Tôn Thất Minh Đăng
Usage:
    # 1. Single pair mode (computes H mapping B -> A and saves H_ref.txt and reference panorama):
    python tools/generate_reference.py dataset/site1/medium/left.jpg dataset/site1/medium/right.jpg -o dataset/site1/medium/H_ref.txt

    # 2. Batch mode over a dataset directory containing sites and resolutions:
    python tools/generate_reference.py --dataset dataset

    # 3. Synthetic pair mode (creates a warped right.jpg and ground truth H_gt.txt from a single photo):
    python tools/generate_reference.py --synthetic dataset/synthetic/medium/left.jpg -o dataset/synthetic/medium
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2


def compute_reference_homography(img_a: np.ndarray, img_b: np.ndarray):
    """Computes homography H mapping points from Image B to Image A (B -> A).

    Uses SIFT (or ORB fallback) and RANSAC to find a highly accurate reference.
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # 1. Feature detection: prefer SIFT for high sub-pixel precision
    try:
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING

    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        raise RuntimeError("Not enough keypoints detected to estimate homography.")

    # 2. Matching with Lowe's ratio test
    matcher = cv2.BFMatcher(norm)
    knn_matches = matcher.knnMatch(desc_b, desc_a, k=2)  # query = B, train = A

    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise RuntimeError(f"Too few good matches found ({len(good_matches)}) to compute homography.")

    # Extract matching coordinates: pts_b -> pts_a (maps B into A)
    pts_b = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 3. Robust RANSAC fitting
    H, inlier_mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    num_inliers = int(np.sum(inlier_mask))
    print(f"  Matches: {len(good_matches)}, RANSAC Inliers: {num_inliers} ({num_inliers/len(good_matches)*100:.1f}%)")

    # Normalize H so H[2, 2] == 1.0
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    return H, num_inliers


def render_reference_panorama(img_a: np.ndarray, img_b: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warps image B into A's frame using H and creates a reference panorama."""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Corners of B in B's coordinate system
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    warped_corners_b = cv2.perspectiveTransform(corners_b, H).reshape(-1, 2)

    # Corners of A
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]])

    all_corners = np.vstack((corners_a, warped_corners_b))
    min_x = np.floor(np.min(all_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(all_corners[:, 1])).astype(int)
    max_x = np.ceil(np.max(all_corners[:, 0])).astype(int)
    max_y = np.ceil(np.max(all_corners[:, 1])).astype(int)

    # Canvas dimensions and translation
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    ox = -min_x
    oy = -min_y

    translation = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)

    # Warp B onto canvas
    canvas = cv2.warpPerspective(img_b, translation @ H, (canvas_w, canvas_h))

    # Place image A onto canvas
    # Create mask for A to overlay or average in the overlap
    mask_a = (img_a.sum(axis=2) > 0).astype(np.uint8)
    a_region = canvas[oy : oy + h_a, ox : ox + w_a]
    canvas[oy : oy + h_a, ox : ox + w_a] = np.where(mask_a[:, :, None], img_a, a_region)

    return canvas


def save_homography_txt(path: Path, H: np.ndarray):
    """Saves H in 3 rows of 3 numbers matching loadH in src/main.cu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in range(3):
            f.write(f"{H[r, 0]:.10g} {H[r, 1]:.10g} {H[r, 2]:.10g}\n")
    print(f"  Saved homography: {path}")


def generate_synthetic_pair(left_path: Path, out_dir: Path):
    """Generates a synthetic right image by applying a known camera rotation homography."""
    img_left = cv2.imread(str(left_path))
    if img_left is None:
        sys.exit(f"Failed to read image: {left_path}")

    h, w = img_left.shape[:2]

    # Create a realistic in-plane camera rotation + slight perspective tilt
    # Center translation, rotate by ~5 degrees, slight zoom and projective tilt
    cx, cy = w / 2.0, h / 2.0
    angle_deg = 3.5
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Affine rotation/scale
    scale = 0.98
    H_rot = np.array([
        [scale * cos_t, -scale * sin_t, cx * (1 - scale * cos_t) + cy * scale * sin_t + w * 0.08],
        [scale * sin_t,  scale * cos_t, cy * (1 - scale * cos_t) - cx * scale * sin_t],
        [0.00003, 0.00001, 1.0]  # Slight projective foreshortening
    ], dtype=np.float64)

    # Right image is formed by warping left with H_rot (maps left -> right, so H_gt for B->A is inverse)
    img_right = cv2.warpPerspective(img_left, H_rot, (w, h))

    # In our project convention, H maps B (right) -> A (left):
    H_gt = np.linalg.inv(H_rot)
    H_gt = H_gt / H_gt[2, 2]

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "left.jpg"), img_left)
    cv2.imwrite(str(out_dir / "right.jpg"), img_right)
    save_homography_txt(out_dir / "H_gt.txt", H_gt)
    print(f"Generated synthetic pair and H_gt.txt in {out_dir}")


def process_single_pair(left_path: Path, right_path: Path, out_h: Path, save_panorama: bool = True):
    print(f"Processing pair:\n  Left:  {left_path}\n  Right: {right_path}")
    img_a = cv2.imread(str(left_path))
    img_b = cv2.imread(str(right_path))
    if img_a is None or img_b is None:
        sys.exit(f"Failed to load images from {left_path} or {right_path}")

    H, inliers = compute_reference_homography(img_a, img_b)
    save_homography_txt(out_h, H)

    if save_panorama:
        pano = render_reference_panorama(img_a, img_b, H)
        pano_path = out_h.parent / f"{out_h.stem}_panorama.jpg"
        cv2.imwrite(str(pano_path), pano)
        print(f"  Saved reference panorama: {pano_path}")


def process_dataset(dataset_dir: Path):
    sites = [p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "synthetic"]
    if not sites:
        sys.exit(f"No sites found in {dataset_dir}")

    for site in sorted(sites):
        print(f"\n--- Site: {site.name} ---")
        for size in ("small", "medium", "large"):
            size_dir = site / size
            if not size_dir.is_dir():
                continue
            left_matches = list(size_dir.glob("left.*"))
            right_matches = list(size_dir.glob("right.*"))
            if not left_matches or not right_matches:
                continue
            left = left_matches[0]
            right = right_matches[0]
            out_h = size_dir / "H_ref.txt"
            try:
                process_single_pair(left, right, out_h, save_panorama=(size == "medium"))
            except Exception as e:
                print(f"  Warning: failed on {site.name}/{size}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", nargs="?", type=Path, help="Path to left image (Image A)")
    parser.add_argument("right", nargs="?", type=Path, help="Path to right image (Image B)")
    parser.add_argument("-o", "--out", type=Path, default=Path("H_ref.txt"), help="Output path for H_ref.txt or synthetic output dir")
    parser.add_argument("--dataset", type=Path, help="Process all site folders in the dataset directory")
    parser.add_argument("--synthetic", type=Path, help="Generate a synthetic pair and H_gt.txt from this source image")

    args = parser.parse_args()

    if args.synthetic:
        out_dir = args.out if args.out != Path("H_ref.txt") else Path("dataset/synthetic/medium")
        generate_synthetic_pair(args.synthetic, out_dir)
    elif args.dataset:
        process_dataset(args.dataset)
    elif args.left and args.right:
        process_single_pair(args.left, args.right, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
