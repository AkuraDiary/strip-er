#!/usr/bin/env python3
"""
Straighten a skewed even/odd export from unstitch.py.

Two modes:
  --mode quad    (default) find the 4 corners of the content and warp them to a
                 rectangle. Fixes parallelogram skew (shear), not just rotation.
  --mode rotate  rotate only, to a minimum-area upright box. Fixes tilt only.

  python normalize.py -i test-img_even.jpeg
  python normalize.py -i test-img_even.jpeg --inset 8
  python normalize.py -i test-img_even.jpeg --mode rotate
  python normalize.py -i clip_even.mp4

Video: the transform is computed once from the first frame and reused for every frame.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def content_hull(frame, thresh=0, open_px=3):
    """Convex hull of the non-black pixels, with speckle removed."""
    mask = (frame.max(axis=2) > thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    pts = cv2.findNonZero(mask)
    if pts is None:
        sys.exit("No non-black content found. Try a lower --thresh.")
    return cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)


def hull_to_quad(hull):
    """
    Reduce the hull to 4 corners by raising the simplification tolerance until
    exactly 4 vertices remain. Falls back to the minimum-area box if that fails.
    """
    peri = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    lo, hi = 0.0, 0.25 * peri
    approx = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), mid, True).reshape(-1, 2)
        if len(approx) > 4:
            lo = mid
        elif len(approx) < 4:
            hi = mid
        else:
            break
    if len(approx) != 4:
        print("warning: could not reduce content to 4 corners, using bounding box fallback")
        approx = cv2.boxPoints(cv2.minAreaRect(hull.reshape(-1, 1, 2)))
    return order_corners(approx.astype(np.float32))


def order_corners(pts):
    """Return corners as top-left, top-right, bottom-right, bottom-left.
    Works for parallelograms: sort by y to split top/bottom, then by x within each pair."""
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bot = pts[2:][np.argsort(pts[2:, 0])]
    tl, tr = top
    bl, br = bot
    return np.array([tl, tr, br, bl], np.float32)


def rect_size(quad):
    tl, tr, br, bl = quad
    d = lambda a, b: float(np.linalg.norm(a - b))
    w = int(round(max(d(tl, tr), d(bl, br))))
    h = int(round(max(d(tl, bl), d(tr, br))))
    return max(w, 1), max(h, 1)


def make_transform(frame, hull, mode):
    """Return (M, out_size) that maps the content into an upright rectangle."""
    if mode == "quad":
        quad = hull_to_quad(hull)
        w, h = rect_size(quad)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        return ("persp", M), (w, h), quad
    return rotate_transform(frame, hull)


def rotate_transform(frame, hull):
    """Rotation-only: minimum-area upright bounding box of the hull."""
    h, w = frame.shape[:2]
    D = int(math.ceil(math.hypot(w, h))) + 2
    cx, cy = w / 2.0, h / 2.0
    sx, sy = D / 2.0 - cx, D / 2.0 - cy

    def transform(a):
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        M[0, 2] += sx
        M[1, 2] += sy
        return M

    def box_for(a):
        p = cv2.transform(hull.reshape(-1, 1, 2), transform(a)).reshape(-1, 2)
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        return (x1 - x0) * (y1 - y0), (x0, y0, x1, y1)

    angles = np.arange(-90.0, 90.0, 0.5)
    a = angles[int(np.argmin([box_for(t)[0] for t in angles]))]
    fine = np.arange(a - 0.5, a + 0.5, 0.02)
    a = fine[int(np.argmin([box_for(t)[0] for t in fine]))]
    _, (x0, y0, x1, y1) = box_for(a)
    # translate so the box starts at (0, 0)
    M = transform(a)
    M[0, 2] -= x0
    M[1, 2] -= y0
    return ("affine", M), (int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))), None


def apply(frame, spec, size, inset):
    kind, M = spec
    w, h = size
    if kind == "persp":
        out = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderValue=(0, 0, 0))
    else:
        out = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=(0, 0, 0))
    if inset > 0:
        if out.shape[1] <= 2 * inset or out.shape[0] <= 2 * inset:
            sys.exit("Inset is larger than the output. Lower --inset.")
        out = out[inset:-inset, inset:-inset]
    return out


def main():
    ap = argparse.ArgumentParser(description="Straighten a skewed strip export.")
    ap.add_argument("-i", "--input", required=True, help="image or video file")
    ap.add_argument("-o", "--output-dir", default=None, help="output folder (default: next to input)")
    ap.add_argument("--mode", choices=["quad", "rotate"], default="quad",
                    help="quad: fix parallelogram skew (default). rotate: rotation only.")
    ap.add_argument("--inset", type=int, default=4,
                    help="pixels trimmed from each edge of the result to remove black border "
                         "and anti-aliased edge pixels (default 4)")
    ap.add_argument("--thresh", type=int, default=0,
                    help="pixel value above which content counts (0 = any non-black)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"File not found: {src}")
    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = src.stem, src.suffix or ".png"

    is_video = src.suffix.lower() in VIDEO_EXT
    cap = cv2.VideoCapture(str(src)) if is_video else None
    if is_video:
        ok, first = cap.read()
        if not ok:
            sys.exit("Could not read the video.")
    else:
        first = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if first is None:
            sys.exit("Could not read the image.")

    hull = content_hull(first, args.thresh)
    spec, size, quad = make_transform(first, hull, args.mode)
    if quad is not None:
        print("corners (TL, TR, BR, BL):", np.round(quad, 1).tolist())
    print(f"output size before inset: {size[0]}x{size[1]}")

    if not is_video:
        out = apply(first, spec, size, args.inset)
        path = out_dir / f"{stem}_straight{ext}"
        cv2.imwrite(str(path), out)
        print(f"saved: {path}  ({out.shape[1]}x{out.shape[0]})")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    path = out_dir / f"{stem}_straight.mp4"
    writer, n, frame = None, 0, first
    while True:
        out = apply(frame, spec, size, args.inset)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (out.shape[1], out.shape[0]))
        writer.write(out)
        n += 1
        if n % 30 == 0:
            print(f"\r  frame {n}/{total}", end="", flush=True)
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    writer.release()
    print(f"\nsaved: {path}  ({n} frames)")


if __name__ == "__main__":
    main()