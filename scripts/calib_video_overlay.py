"""Calibrate the fixed-camera world->pixel homography of render_replay_sync.py videos
(--camera fixed --zoom 62 --size 1080x1080, 32x32 map) from rendered frames + ground-truth unit positions.

Detects unit sprites as non-grass, non-black blobs, pairs isolated units with blobs, fits a homography (DLT)
on blob centroids, iterating assignment <-> fit. usage: python calib_video_overlay.py VIDEO.mp4 UNITS_DUMP.json [...pairs]
"""
import json
import sys

import imageio
import numpy as np
from scipy import ndimage, optimize


def fit_h(w, p):
    A = []
    for (X, Y), (x, y) in zip(w, p):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y, -y])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def cam(q, X, Y):
    """pinhole camera centred on world x=16, looking north with pitch phi from distance D at (16, ty)."""
    f, D, phi, ox, oy, ty = q
    dX, dY = np.asarray(X) - 16.0, np.asarray(Y) - ty
    z = D + dY * np.cos(phi)
    return ox + f * dX / z, oy - f * dY * np.sin(phi) / z


def fit_cam(W, P, q0):
    W, P = np.asarray(W, float), np.asarray(P, float)
    r = lambda q: np.concatenate([cam(q, W[:, 0], W[:, 1])[0] - P[:, 0], cam(q, W[:, 0], W[:, 1])[1] - P[:, 1]])
    return optimize.least_squares(r, q0, loss="soft_l1", f_scale=5).x


def apply(H, X, Y):
    if len(H) == 6:
        return cam(H, X, Y)
    v = H @ np.array([X, Y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def blobs(img, bg):
    d = np.abs(img.astype(float) - bg).sum(-1)
    m = ndimage.binary_dilation(d > 60, iterations=4)
    lab, n = ndimage.label(m)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.nonzero((lab[sl] == i) & (d[sl] > 60))
        if len(xs) < 60:
            continue
        out.append((xs.mean() + sl[1].start, ys.mean() + sl[0].start, len(xs)))
    return out


def main():
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    # rough start: map top corners + three hand-read unit positions from runs/demo3v/render/f82.png
    H = [1765, 50, 1.08, 540, 540, 16]
    obs = []
    for vid, dump in pairs:
        U = json.load(open(dump))["units"]
        frames = [f for i, f in enumerate(imageio.get_reader(vid)) if i % 4 == 0]
        bg = np.median(np.stack(frames[::2]), 0)
        for j, img in enumerate(frames):
            i = 4 * j
            if i >= len(U):
                continue
            us = [u for u in U[i] if u["hp"] > 0]
            bl = blobs(img, bg)
            for u in us:
                if u["type"] in (4, 1971):  # colossi are tall: skip for the ground fit
                    continue
                if any(v is not u and np.hypot(v["x"] - u["x"], v["y"] - u["y"]) < 4.0 for v in us):
                    continue
                obs.append((u["x"], u["y"], bl))
    for it in range(8):
        W, P = [], []
        for X, Y, bl in obs:
            px, py = apply(H, X, Y)
            if not bl:
                continue
            d = [np.hypot(bx - px, by - py) for bx, by, _ in bl]
            k = int(np.argmin(d))
            if d[k] < (90 if it == 0 else 60 if it < 3 else 35) and sum(dd < 60 for dd in d) == 1:
                W.append((X, Y)); P.append(bl[k][:2])
        H = fit_cam(W, P, H if len(H) == 6 else [1765, 50, 1.08, 540, 540, 16])
        res = [np.hypot(*(np.array(apply(H, *w)) - p)) for w, p in zip(W, P)]
        print(f"iter {it}: {len(W)} matches, median residual {np.median(res):.1f}px, p90 {np.percentile(res, 90):.1f}px")
    print(json.dumps(list(map(float, H))))
    for X, Y in [(0, 0), (32, 0), (0, 32), (32, 32), (16, 16), (2, 2), (30, 30)]:
        print((X, Y), [round(v) for v in apply(H, X, Y)])


if __name__ == "__main__":
    main()
