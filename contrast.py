import cv2
import numpy as np
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tkinter import Tk, filedialog
import os
import pandas as pd
import pyautogui


def resize_to_fit_window(img, max_width=1000, max_height=800):
    """
    Rescale `img` so it fits within (max_width, max_height) without upscaling.
    Returns (resized_img, scale_factor).
    """
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def compute_contrast(img_gray, p1, p2):
    """
    Given a grayscale image `img_gray` (float in [0,1]) and two pixel coordinates
    p1=(x1,y1), p2=(x2,y2), extract the intensity profile along that line and compute:
        I_max, I_min, ΔI = I_max - I_min, and Michaelson contrast = (I_max - I_min)/(I_max + I_min)
    """
    r0, c0 = p1[1], p1[0]
    r1, c1 = p2[1], p2[0]
    profile = profile_line(img_gray, (r0, c0), (r1, c1), mode='reflect')
    I_max = profile.max()
    I_min = profile.min()
    contrast_diff = I_max - I_min
    contrast_michaelson = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) != 0 else np.nan
    return I_max, I_min, contrast_diff, contrast_michaelson

def process_image(file_path, annotated_dir):
    """
    Show one image, let the user click as many point-pairs as they like:
      - Each time two points are clicked, draw a line and compute contrast.
      - Store the results in a list of dicts.
      - Press 'n' to finish this image and move on.
      - Press ESC to abort completely (exit all).
    Returns:
      image_results: list of dicts (one per pair)
      exit_all    : True if ESC was pressed (stop everything), False otherwise
      annotated_img: the final annotated (scaled) image (cv2 BGR) with all points/lines drawn
    """
    # Read image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Could not read image: {file_path}")
        return [], False, None

    # Prepare grayscale for contrast calculation (float [0,1])
    gray = rgb2gray(img) if img.shape[2] == 3 else img.astype(float)
    gray = gray.astype(float)
    gray /= gray.max()

    # Resize for display
    img_disp, scale = resize_to_fit_window(img.copy(), max_width=1600, max_height=900)
    annotated = img_disp.copy()  # We'll draw circles/lines here

    # This will hold all pairs for this image
    image_results = []
    current_points = []
    exit_all = False

    # Mouse callback: every time the user clicks, store a point,
    # draw it; once we have 2, draw the line, compute/record contrast, then reset.
    def click_event(event, x, y, flags, param):
        nonlocal current_points, annotated, image_results
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record the click (in display coordinates)
            current_points.append((x, y))
            # Draw a small circle at the clicked location
            cv2.circle(annotated, (x, y), 1, (0, 255, 255), -1)

            if len(current_points) == 2:
                # Draw a line between the two clicked points
                cv2.line(annotated, current_points[0], current_points[1], (0, 255, 255), 1)

                # Convert display coords → original image coords
                p1 = (int(current_points[0][0] / scale), int(current_points[0][1] / scale))
                p2 = (int(current_points[1][0] / scale), int(current_points[1][1] / scale))
                I_max, I_min, contrast_diff, contrast_michaelson = compute_contrast(gray, p1, p2)

                # Record everything in a dict
                image_results.append({
                    "filename": file_path,
                    "x1": p1[0], "y1": p1[1],
                    "x2": p2[0], "y2": p2[1],
                    "I_max": I_max,
                    "I_min": I_min,
                    "ΔI": contrast_diff,
                    "Michaelson": contrast_michaelson
                })

                # Clear for the next pair
                current_points = []

            # Refresh display window
            cv2.imshow(window_name, annotated)

    window_name = "Select pairs (press 'n'→next image, ESC→exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated)
    cv2.setMouseCallback(window_name, click_event)

    print("  - Click two points for each contrast measurement.")
    print("  - Repeat as many times as needed.")
    print("  - Press 'n' when you want to move to the next image.")
    print("  - Press ESC at any time to exit and save.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            # Done with this image; move on
            break
        elif key == 27:  # ESC
            exit_all = True
            break

    cv2.destroyAllWindows()

    # Save the annotated image even if no pairs were clicked (so user knows which images were shown)
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    annotated_path = os.path.join(annotated_dir, f"{name}_annotated.png")
    cv2.imwrite(annotated_path, annotated)

    return image_results, exit_all, annotated
