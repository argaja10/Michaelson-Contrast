import os
import cv2
import numpy as np
import pyautogui
from tkinter import Tk, filedialog
from contrast import resize_to_fit_window


def load_image_channel(filepath: str, flag: int = 0):
    """Load image and extract the requested channel (0: BGR, 1: R, 2: G, 3: B)."""
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    b, g, r = cv2.split(img)
    channels = {0: img, 1: r, 2: g, 3: b}
    return channels.get(flag, img)


def prepare_display(image: np.ndarray):
    """Resize to screen, convert to BGR for drawing, and return (display_img, scale)."""
    h, w = image.shape[:2]
    sw, sh = pyautogui.size()
    disp, scale = resize_to_fit_window(image.copy(), max_width=sw, max_height=sh)
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    return disp, scale


def select_points_or_roi(window_name: str, disp: np.ndarray, 
                         mode_keys: dict, on_draw):
    """
    Generic interactive selector:
      - mode_keys: {'f': 'foreground', 'b': 'background', ...}
      - on_draw: callback to redraw disp with current points/ROI
    Returns collected data and whether ESC was pressed.
    """
    mode = [next(iter(mode_keys.values()))]  # default mode
    stop = False

    def mouse_cb(event, x, y, flags, param):
        on_draw(event, x, y, flags, param, mode[0])
        on_draw(None, None, None, None, None, mode[0])  # trigger redraw

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_cb)
    on_draw(None, None, None, None, None, mode[0])  # initial draw

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            stop = True
            break
        for k, m in mode_keys.items():
            if key == ord(k):
                mode[0] = m
        if key == ord('n'):
            break

    cv2.destroyWindow(window_name)
    return stop


def compute_snr_from_points(gray: np.ndarray, fg_pts, bg_pts, eps=1e-12):
    fg = np.array(fg_pts, dtype=int)
    bg = np.array(bg_pts, dtype=int)
    sig = gray[fg[:,1], fg[:,0]].astype(np.float64)
    noise = gray[bg[:,1], bg[:,0]].astype(np.float64)
    P_signal = np.mean(sig**2)
    P_noise = noise.var(ddof=0)
    snr_db = 10 * np.log10(P_signal / (P_noise + eps))
    return snr_db, P_signal, P_noise


def compute_snr_from_roi(gray: np.ndarray, fg_pts, bg_poly, eps=1e-12):
    fg = np.array(fg_pts, dtype=int)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(bg_poly)], color=255)
    bg_vals = gray[mask == 255].astype(np.float64)

    mu_signal = gray[fg[:,1], fg[:,0]].mean()
    mu_bg = bg_vals.mean()
    sigma_bg = bg_vals.std(ddof=0)
    snr = (mu_signal - mu_bg) / (sigma_bg + eps)
    return snr, mu_signal, mu_bg, sigma_bg


def annotate_and_save(out_dir, filename, disp, fg_pts, bg_shape, closed=False):
    """Draw final annotations onto disp and save to out_dir."""
    img = disp.copy()
    for x, y in fg_pts:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    if "points" in out_dir:
        for x, y in bg_shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    else:
        pts = np.array(bg_shape)
        cv2.polylines(img, [pts], isClosed=closed, color=(0, 0, 255), thickness=2)
        
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, img)


def process_image_snr_points(filepath, annotated_dir, flag=0, results=None):
    """Point‐based SNR: user clicks fg and bg points."""
    if results is None:
        results = []

    fname = os.path.basename(filepath)
    img_chan = load_image_channel(filepath, flag)
    disp, scale = prepare_display(img_chan)
    fg_pts, bg_pts = [], []

    def on_draw(event, x, y, flags, param, mode):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = (x, y)
            if mode == 'foreground':
                fg_pts.append(pt)
            elif mode == 'background':
                bg_pts.append(pt)
        temp = disp.copy()
        for p in fg_pts:
            cv2.circle(temp, p, 1, (0,255,0), -1)
        for p in bg_pts:
            cv2.circle(temp, p, 1, (0,0,255), -1)
        cv2.imshow("Select Points (f/b: mode, n: next, ESC)", temp)

    stop = select_points_or_roi(
        "Select Points (f/b: mode, n: next, ESC)",
        disp, {'f':'foreground','b':'background'}, on_draw
    )
    if stop or not fg_pts or not bg_pts:
        print(f"Skipping {fname}: insufficient data")
        return results, stop

    gray = cv2.cvtColor(img_chan, cv2.COLOR_BGR2GRAY) if img_chan.ndim==3 else img_chan
    snr_db, P_sig, P_noise = compute_snr_from_points(gray, fg_pts, bg_pts)
    annotate_and_save(annotated_dir, fname, disp, fg_pts, bg_pts, closed=False)

    results.append({
        'filename': fname,
        'num_fg': len(fg_pts),
        'num_bg': len(bg_pts),
        'snr_db': snr_db,
    })
    return results, stop


def process_image_snr_roi(filepath, annotated_dir, flag=0, results=None):
    """ROI‐based SNR: user marks fg points and freehand bg polygon."""
    if results is None:
        results = []

    fname = os.path.basename(filepath)
    img_chan = load_image_channel(filepath, flag)
    disp, scale = prepare_display(img_chan)
    fg_pts, bg_poly = [], []
    drawing = [False]

    def on_draw(event, x, y, flags, param, mode):
        if mode == 'foreground' and event == cv2.EVENT_LBUTTONDOWN:
            fg_pts.append((x,y))
        elif mode == 'background':
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing[0] = True; bg_poly.append((x,y))
            elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
                bg_poly.append((x,y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing[0] = False
        temp = disp.copy()
        for p in fg_pts:
            cv2.circle(temp, p, 1, (0,255,0), -1)
        if len(bg_poly)>1:
            cv2.polylines(temp, [np.array(bg_poly)], isClosed=False, color=(0,0,255), thickness=1)
        cv2.imshow("Draw ROI (f/b: mode, n: next, ESC)", temp)

    stop = select_points_or_roi(
        "Draw ROI (f/b: mode, n: next, ESC)",
        disp, {'f':'foreground','b':'background'}, on_draw
    )
    if stop or not fg_pts or len(bg_poly)<3:
        print(f"Skipping {fname}: insufficient data")
        return results, stop

    # unscale points
    fg = [(int(x/scale), int(y/scale)) for x,y in fg_pts]
    bg = [(int(x/scale), int(y/scale)) for x,y in bg_poly]
    gray = cv2.cvtColor(img_chan, cv2.COLOR_BGR2GRAY) if img_chan.ndim==3 else img_chan
    snr, mu_s, mu_b, sigma_b = compute_snr_from_roi(gray, fg, bg)
    annotate_and_save(annotated_dir, fname, disp, fg_pts, bg_poly, closed=True)

    results.append({
        'filename': fname,
        'num_fg': len(fg_pts),
        'poly_length': len(bg_poly),
        'snr': snr,
    })
    return results, stop
