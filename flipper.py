#!/usr/bin/env python3
"""
Diagonal strip un-interleaver: interactive tuner and exporter.

  python unstitch.py -i input.png
  python unstitch.py -i input.mp4 -o out/ --pos 0.3

Keys (click the window first):
  A / D      angle -/+ 0.05          J / L      angle -/+ 0.005
  W / E      width -/+ 0.02          I / K      width -/+ 0.002
  Z / C      phase -/+ 0.5% period   U / P      duty  -/+ 0.5%
  G          guide: off / comb / full strips
  O          solid / translucent comb
  [ / ]      comb band thickness
  R          re-estimate geometry from the image
  , / .      video: seek -/+ 1 second
  S          export full resolution
  Q / ESC    quit

Mouse: drag over the left panel to move the comb band.
Wheel: width. Shift+wheel: angle. Ctrl+wheel: phase.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
WIN = "Strip Tuner"
PANEL_MAX = 480      # max width of each preview panel (px)
HUD_H = 60
RED = (30, 20, 225)  # BGR
CYAN = (255, 255, 0)


def wrap_angle(a):
    """Map any angle into [-90, 90)."""
    return (a + 90.0) % 180.0 - 90.0


# ------------------------------------------------------------ remap maps
def build_deinterlace_maps(w, h, line_angle_deg, period, duty_ratio, phase_offset):
    """
    Remap maps for the two media.
    period: strip A + strip B width in px.
    duty_ratio: fraction of the period occupied by strip A.
    phase_offset: where strip A starts, measured from the image center.
    """
    normal = math.radians((line_angle_deg + 90.0) % 360.0)
    nx, ny = np.float32(math.cos(normal)), np.float32(math.sin(normal))
    tx, ty = -ny, nx

    cx, cy = w * 0.5, h * 0.5
    x_1d = np.arange(w, dtype=np.float32) - cx
    y_1d = np.arange(h, dtype=np.float32) - cy
    u_out = np.add.outer(y_1d * ny, x_1d * nx)
    v_out = np.add.outer(y_1d * ty, x_1d * tx)

    w_even = period * duty_ratio
    w_odd = period * (1.0 - duty_ratio)

    u_rel = u_out - phase_offset
    k_e = np.floor(u_rel / w_even)
    u_src_even = phase_offset + k_e * period + (u_rel - k_e * w_even)

    k_o = np.floor(u_rel / w_odd)
    u_src_odd = phase_offset + k_o * period + w_even + (u_rel - k_o * w_odd)

    def to_xy(u_src):
        return ((u_src * nx + v_out * tx + cx).astype(np.float32),
                (u_src * ny + v_out * ty + cy).astype(np.float32))

    return to_xy(u_src_even), to_xy(u_src_odd)

def _strip_u(x, sw, period, base, kmin):
    """Source u-coordinate for packed column x of one parity."""
    k = np.floor(x / sw)
    return base + (k + kmin) * period + (x - k * sw)


def pack_maps(w, h, angle, period, duty, phase, parity):
    """
    Remap maps that lay one parity's strips side by side on an upright canvas.
    Output is a rectangle: columns = strips of this parity in order,
    rows = full tangent extent of the source. Nothing is clipped.
    """
    th = math.radians(angle + 90.0)
    nx, ny = math.cos(th), math.sin(th)
    tx, ty = -ny, nx
    cx, cy = w * 0.5, h * 0.5

    us, vs = [], []
    for dx, dy in ((-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)):
        us.append(dx * nx + dy * ny)
        vs.append(dx * tx + dy * ty)
    umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)

    if parity == 0:
        sw, base = period * duty, phase
    else:
        sw, base = period * (1.0 - duty), phase + period * duty

    kmin = int(math.floor((umin - base) / period)) - 1
    kmax = int(math.ceil((umax - base) / period)) + 1
    n = int(math.ceil((kmax - kmin + 1) * sw)) + 1
    x = np.arange(n, dtype=np.float64)
    u = _strip_u(x, sw, period, base, kmin)

    # keep only columns that fall inside the source (u is monotonic, so this is contiguous)
    valid = np.where((u >= umin) & (u <= umax))[0]
    if len(valid) == 0:
        sys.exit("A medium came out empty. Check the angle, period and phase.")
    u = u[valid[0]:valid[-1] + 1]

    v = vmin + np.arange(int(math.ceil(vmax - vmin)), dtype=np.float64)
    mx = np.add.outer(v * tx, u * nx) + cx
    my = np.add.outer(v * ty, u * ny) + cy
    return mx.astype(np.float32), my.astype(np.float32)


def upright_maps(Wp, Hp, angle):
    """
    Maps that rotate a packed (Wp x Hp) rectangle back to the original orientation,
    cropped to the largest centered rectangle (same aspect) with no black corners.
    """
    th = math.radians(angle + 90.0)
    nx, ny = math.cos(th), math.sin(th)
    tx, ty = -ny, nx

    def fits(a, b):
        for X, Y in ((-a, -b), (a, -b), (-a, b), (a, b)):
            lx = X * nx + Y * ny
            ly = X * tx + Y * ty
            if abs(lx) > Wp / 2 + 1e-6 or abs(ly) > Hp / 2 + 1e-6:
                return False
        return True

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits(mid * Wp / 2, mid * Hp / 2):
            lo = mid
        else:
            hi = mid
    ow = max(2, int(2 * lo * Wp / 2))
    oh = max(2, int(2 * lo * Hp / 2))

    X = np.arange(ow, dtype=np.float64) - ow / 2 + 0.5
    Y = np.arange(oh, dtype=np.float64) - oh / 2 + 0.5
    mx = np.add.outer(Y * ny, X * nx) + Wp / 2
    my = np.add.outer(Y * ty, X * tx) + Hp / 2
    return mx.astype(np.float32), my.astype(np.float32)


def render(frame, plan_entry, interp):
    (mx, my), up = plan_entry
    out = cv2.remap(frame, mx, my, interp, borderValue=(0, 0, 0))
    if up is not None:
        out = cv2.remap(out, up[0], up[1], interp, borderValue=(0, 0, 0))
    return out


def plan_size(plan_entry):
    (mx, my), up = plan_entry
    m = up[0] if up is not None else mx
    return (m.shape[1], m.shape[0])
# ------------------------------------------------------------ global estimate
def estimate_geometry(bgr, max_dim=768):
    """
    Global fit of strip angle, boundary spacing and boundary phase over the whole image.
    Returns (line_angle_deg, strip_width_px, boundary_offset_px) in original pixels.
    boundary_offset is relative to the image center; take it modulo the period (2 * width).
    """
    h0, w0 = bgr.shape[:2]
    s = min(1.0, max_dim / max(h0, w0))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy).ravel()

    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xs = xs.ravel() - W * 0.5
    ys = ys.ravel() - H * 0.5

    diag = math.hypot(H, W) * 0.5
    BIN = 0.5
    nb = int(2 * diag / BIN) + 1
    centers = -diag + (np.arange(nb) + 0.5) * BIN

    def profile(theta_deg):
        t = math.radians(theta_deg)
        u = xs * math.cos(t) + ys * math.sin(t)
        idx = np.clip(np.floor((u + diag) / BIN).astype(np.int64), 0, nb - 1)
        return np.bincount(idx, weights=g, minlength=nb)

    def search(thetas, spacings):
        E = np.exp(-2j * np.pi * np.outer(1.0 / spacings, centers))
        best = (-1.0, 0.0, 0.0, 0.0)
        for th in thetas:
            prof = profile(th)
            S = E @ prof
            sc = np.abs(S) / (prof.sum() + 1e-9)
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (sc[i], th, spacings[i], -np.angle(S[i]) / (2 * np.pi) * spacings[i])
        return best

    # coarse: normal angle every 1 degree, boundary spacing 3 .. min/4 px
    _, th0, _, _ = search(np.arange(0, 180, 1.0),
                          np.linspace(3, max(3.5, min(H, W) / 4), 400))
    # fine
    _, th, sp, ph = search(np.arange(th0 - 1.0, th0 + 1.0, 0.02),
                           np.linspace(0, 1, 1))  # placeholder, replaced below
    sp_fine = None
    # refine spacing around coarse winner
    _, _, sp_c, _ = search(np.array([th0]), np.linspace(3, max(3.5, min(H, W) / 4), 400))
    _, th, sp, ph = search(np.arange(th0 - 1.0, th0 + 1.0, 0.02),
                           np.linspace(sp_c * 0.97, sp_c * 1.03, 300))

    line_angle = wrap_angle(th - 90.0)
    return line_angle, sp / s, ph / s


# ------------------------------------------------------------ tuner
class Tuner:
    def __init__(self, args):
        self.args = args
        self.input = Path(args.input)
        self.is_video = self.input.suffix.lower() in VIDEO_EXT
        self.cap = None
        self.total = 1
        self.fps = 30.0
        self.frame_idx = 0

        if self.is_video:
            self.cap = cv2.VideoCapture(str(self.input))
            if not self.cap.isOpened():
                sys.exit("Could not open video.")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.frame_idx = int(np.clip(args.pos, 0, 1) * (self.total - 1))
            frame = self.read_at(self.frame_idx)
        else:
            frame = cv2.imread(str(self.input), cv2.IMREAD_COLOR)
        if frame is None:
            sys.exit("Could not read a frame from the input.")

        self.h, self.w = frame.shape[:2]
        self.s = min(1.0, PANEL_MAX / max(self.w, self.h))
        self.pw = max(64, round(self.w * self.s))
        self.ph = max(64, round(self.h * self.s))
        self.band_half = min(self.pw, self.ph) * 0.16

        self.angle = 0.0
        self.width = 8.0
        self.phase = 0.0
        self.duty = 0.5
        self.band_pos = 0.5
        self.solid = True
        self.guide = 1   # 0 off, 1 comb swatch, 2 full strips

        self.dragging = False
        self.dirty = True
        self.redraw = True
        self.set_frame(frame)

    # --- frames
    def read_at(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        return f if ok else None

    def set_frame(self, frame):
        self.frame = frame
        self.prev_src = cv2.resize(frame, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        self.dirty = True
        self.redraw = True

    # --- parameters
    @property
    def period(self):
        return 2.0 * self.width

    def set_angle(self, a):
        self.angle = wrap_angle(a); self.dirty = True

    def set_width(self, w):
        self.width = max(0.5, w); self.dirty = True

    def set_phase(self, p):
        self.phase = p % self.period; self.dirty = True

    def set_duty(self, d):
        self.duty = float(np.clip(d, 0.1, 0.9)); self.dirty = True

    def apply_estimate(self):
        ang, sp, ph = estimate_geometry(self.frame)
        self.angle = ang
        self.width = sp
        self.phase = ph % (2 * sp)
        self.dirty = True
        self.redraw = True
        print(f"estimated: angle={ang:.3f}  width={sp:.3f}px  period={2*sp:.3f}px  phase={self.phase:.2f}px")

    # --- rendering
    def rebuild(self):
        """Recompute the even and odd previews. Runs only when parameters change."""
        per = self.period * self.s
        ph = self.phase * self.s
        (mxe, mye), (mxo, myo) = build_deinterlace_maps(
            self.pw, self.ph, self.angle, per, self.duty, ph)
        self.even_prev = cv2.remap(self.prev_src, mxe, mye, cv2.INTER_LINEAR)
        self.odd_prev = cv2.remap(self.prev_src, mxo, myo, cv2.INTER_LINEAR)
        self.dirty = False

    def comb_panel(self):
        """Left panel: source with the red comb overlay. Runs on every mouse move."""
        base = self.prev_src.copy()
        if self.guide == 0:
            return base

        per = self.period * self.s
        ph = self.phase * self.s
        w_even = per * self.duty
        th = math.radians(self.angle + 90.0)
        nx, ny = math.cos(th), math.sin(th)
        tx, ty = -ny, nx
        cx, cy = self.pw * 0.5, self.ph * 0.5

        corners = [(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)]
        uc = [dx * nx + dy * ny for dx, dy in corners]
        vc = [dx * tx + dy * ty for dx, dy in corners]
        umin, umax = min(uc) - 8, max(uc) + 8
        vmin, vmax = min(vc), max(vc)

        if self.guide == 1:
            vcen = vmin + self.band_pos * (vmax - vmin)
            v0, v1 = vcen - self.band_half, vcen + self.band_half
        else:
            v0, v1 = vmin - 8, vmax + 8

        kmin = int(math.floor((umin - ph) / per)) - 1
        kmax = int(math.ceil((umax - ph) / per)) + 1
        polys = []
        for k in range(kmin, kmax + 1):
            u0 = ph + k * per
            u1 = u0 + w_even
            pts = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            polys.append(np.array([[cx + u * nx + v * tx, cy + u * ny + v * ty]
                                   for u, v in pts], np.int32))

        if self.solid:
            cv2.fillPoly(base, polys, RED)
        else:
            overlay = base.copy()
            cv2.fillPoly(overlay, polys, RED)
            cv2.addWeighted(overlay, 0.6, base, 0.4, 0, dst=base)

        vm = 0.5 * (v0 + v1)
        px = int(round(cx + ph * nx + vm * tx))
        py = int(round(cy + ph * ny + vm * ty))
        cv2.circle(base, (px, py), 3, CYAN, -1, cv2.LINE_AA)
        return base

    def compose(self):
        comb = self.comb_panel()
        div = np.full((self.ph, 2, 3), 90, np.uint8)
        row = np.hstack([comb, div, self.even_prev, div, self.odd_prev])

        hud = np.zeros((HUD_H, row.shape[1], 3), np.uint8)
        frame_txt = f"  frame {self.frame_idx}/{self.total - 1}" if self.is_video else ""
        line1 = (f"angle {self.angle:.3f}  width {self.width:.3f}px  period {self.period:.3f}px  "
                 f"phase {self.phase:.2f}px  duty {self.duty*100:.1f}%  "
                 f"{'SOLID' if self.solid else 'CLEAR'}{frame_txt}")
        line2 = ("[A/D] angle  [J/L] angle fine  [W/E] width  [I/K] width fine  [Z/C] phase  "
                 "[U/P] duty  [G] guide  [R] re-estimate  [S] export packed  [X] export upright  [Q] quit")
        cv2.putText(hud, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        return np.vstack([hud, row])

    # --- input
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and x < self.pw:
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.band_pos = float(np.clip((y - HUD_H) / self.ph, 0.0, 1.0))
            self.redraw = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            step = 1 if cv2.getMouseWheelDelta(flags) > 0 else -1
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.set_angle(self.angle + step * 0.02)
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                self.set_phase(self.phase + step * self.period * 0.01)
            else:
                self.set_width(self.width + step * 0.005)
            self.redraw = True

    def handle_key(self, key):
        ch = chr(key).lower() if key < 128 else ""
        P = self.period
        if ch == "a":   self.set_angle(self.angle - 0.05)
        elif ch == "d": self.set_angle(self.angle + 0.05)
        elif ch == "j": self.set_angle(self.angle - 0.005)
        elif ch == "l": self.set_angle(self.angle + 0.005)
        elif ch == "w": self.set_width(self.width - 0.02)
        elif ch == "e": self.set_width(self.width + 0.02)
        elif ch == "i": self.set_width(self.width - 0.002)
        elif ch == "k": self.set_width(self.width + 0.002)
        elif ch == "z": self.set_phase(self.phase - P * 0.005)
        elif ch == "c": self.set_phase(self.phase + P * 0.005)
        elif ch == "u": self.set_duty(self.duty - 0.005)
        elif ch == "p": self.set_duty(self.duty + 0.005)
        elif ch == "g": self.guide = (self.guide + 1) % 3
        elif ch == "o": self.solid = not self.solid
        elif ch == "[": self.band_half = max(5.0, self.band_half - 4.0)
        elif ch == "]": self.band_half = min(max(self.pw, self.ph), self.band_half + 4.0)
        elif ch == "r": self.apply_estimate()
        elif ch == "s": self.export(upright=False)
        elif ch == "x": self.export(upright=True)
        elif ch in (",", ".") and self.is_video:
            delta = -int(round(self.fps)) if ch == "," else int(round(self.fps))
            self.frame_idx = int(np.clip(self.frame_idx + delta, 0, self.total - 1))
            f = self.read_at(self.frame_idx)
            if f is not None:
                self.set_frame(f)
        self.redraw = True

    def build_plan(self, w, h, upright):
        plan = []
        for parity in (0, 1):
            mx, my = pack_maps(w, h, self.angle, self.period, self.duty, self.phase, parity)
            up = upright_maps(mx.shape[1], mx.shape[0], self.angle) if upright else None
            plan.append(((mx, my), up))
        return plan

    def export(self, upright=False):
        out = Path(self.args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = "upright" if upright else "packed"
        stem = self.input.stem
        names = [f"{stem}_{tag}_even", f"{stem}_{tag}_odd"]

        if not self.is_video:
            plan = self.build_plan(self.w, self.h, upright)
            paths = []
            for p in (0, 1):
                img = render(self.frame, plan[p], cv2.INTER_CUBIC)
                path = out / (names[p] + ".png")
                cv2.imwrite(str(path), img)
                paths.append(path)
            print("saved:\n  " + "\n  ".join(str(p) for p in paths))
            return

        plan = self.build_plan(self.w, self.h, upright)
        cap = cv2.VideoCapture(str(self.input))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        paths = [out / (n + ".mp4") for n in names]
        writers = [cv2.VideoWriter(str(paths[p]), fourcc, fps, plan_size(plan[p]))
                   for p in (0, 1)]

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for p in (0, 1):
                writers[p].write(render(frame, plan[p], cv2.INTER_LINEAR))
            n += 1
            if n % 30 == 0:
                print(f"\r  frame {n}/{self.total}", end="", flush=True)
        cap.release()
        for wr in writers:
            wr.release()
        print("\nsaved:\n  " + "\n  ".join(str(p) for p in paths))
    # --- main loop
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.apply_estimate()

        while True:
            if self.dirty:
                self.rebuild()
                self.redraw = True
            if self.redraw:
                cv2.imshow(WIN, self.compose())
                self.redraw = False

            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                continue
            if key in (27, ord("q"), ord("Q")):
                break
            self.handle_key(key)

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Diagonal strip un-interleaver")
    ap.add_argument("-i", "--input", required=True, help="image or video file")
    ap.add_argument("-o", "--output-dir", default="./output", help="export folder")
    ap.add_argument("--pos", type=float, default=0.5,
                    help="video only: relative position of the sample frame (0..1)")
    args = ap.parse_args()

    if not Path(args.input).is_file():
        sys.exit(f"File not found: {args.input}")

    Tuner(args).run()


if __name__ == "__main__":
    main()