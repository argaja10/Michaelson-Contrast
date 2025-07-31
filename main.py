import sys
from contrast import compute_contrast, resize_to_fit_window, process_image  # Assuming you have a contrast module
import cv2
import numpy as np
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tkinter import Tk, filedialog
import os
import pandas as pd
#from snr import compute_snr_from_points  # Assuming you have a snr module
import pyautogui
from snr import process_image_snr_roi, process_image_snr_points  # Assuming you have a snr module for ROI processing

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QComboBox, QMessageBox, QRadioButton, QButtonGroup, QCheckBox
)

class ImageMetricApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Metric Selector")
        self.setGeometry(100, 100, 300, 250)

        # Label
        self.label = QLabel("Choose computation type:", self)

        # Combo Box
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["Contrast Computation", "SNR Computation"])
        self.combo_box.currentTextChanged.connect(self.update_ui)

        # Radio Buttons
        self.radio_button_roi = QRadioButton("ROI Selection")
        self.radio_button_pt = QRadioButton("Point Selection")
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_button_roi)
        self.radio_group.addButton(self.radio_button_pt)
        self.radio_button_roi.hide()
        self.radio_button_pt.hide()

        # Checkboxes (mutually exclusive)
        self.checkbox_red = QCheckBox("Red Channel", self)
        self.checkbox_green = QCheckBox("Green Channel", self)
        self.checkbox_blue = QCheckBox("Blue Channel", self)
        self.checkbox_rgb = QCheckBox("RGB Channels", self)

        for cb in [self.checkbox_red, self.checkbox_green, self.checkbox_blue, self.checkbox_rgb]:
            cb.stateChanged.connect(lambda state, checked_cb=cb: self.enforce_single_checkbox(checked_cb))
            cb.hide()

        # OK Button
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.handle_ok)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo_box)
        self.layout.addWidget(self.radio_button_roi)
        self.layout.addWidget(self.radio_button_pt)
        self.layout.addWidget(self.checkbox_red)
        self.layout.addWidget(self.checkbox_green)
        self.layout.addWidget(self.checkbox_blue)
        self.layout.addWidget(self.checkbox_rgb)
        self.layout.addWidget(self.ok_button)
        self.setLayout(self.layout)

    def update_ui(self, text):
        show = text == "SNR Computation"
        self.radio_button_roi.setVisible(show)
        self.radio_button_pt.setVisible(show)
        self.checkbox_red.setVisible(show)
        self.checkbox_green.setVisible(show)
        self.checkbox_blue.setVisible(show)
        self.checkbox_rgb.setVisible(show)

    def enforce_single_checkbox(self, selected_checkbox):
        """Ensure only one checkbox is selected at a time."""
        if selected_checkbox.isChecked():
            for cb in [self.checkbox_red, self.checkbox_green, self.checkbox_blue, self.checkbox_rgb]:
                if cb != selected_checkbox:
                    cb.setChecked(False)

    def handle_ok(self):
        choice = self.combo_box.currentText()

        if choice == "Contrast Computation":
            self.compute_contrast()
            return
        elif choice == "SNR Computation":
            # --- For SNR Computation ---
            checkbox_to_flag = {
                self.checkbox_red: 1,
                self.checkbox_green: 2,
                self.checkbox_blue: 3,
                self.checkbox_rgb: 0  # RGB as default flag=0
            }

            checked = [(box, val) for box, val in checkbox_to_flag.items() if box.isChecked()]
            if not checked:
                QMessageBox.warning(self, "Warning", "Please select exactly one color channel.")
                return
            flag = checked[0][1]

            if self.radio_button_roi.isChecked():
                print("SNR with ROI selected")
                self.compute_snr_with_roi(flag)
            elif self.radio_button_pt.isChecked():
                print("SNR with Points selected")
                self.compute_snr_with_points(flag)
            else:
                QMessageBox.warning(self, "Warning", "Please select ROI or Point mode.")

        
    def compute_contrast(self):
        # Replace with your actual contrast logic
        QMessageBox.information(self, "Contrast", "Running contrast computation...")
        Tk().withdraw()
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder or not os.path.exists(folder):
            print("Invalid folder path.")
            return

        # Gather all image files in that folder
        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(image_extensions)
        ]
        if not image_files:
            print("No images found in the selected folder.")
            return

        # Prepare an "annotated" subfolder to save annotated images
        annotated_dir = os.path.join(folder, "annotated-contrast")
        os.makedirs(annotated_dir, exist_ok=True)

        # This will accumulate every pair result from all images
        all_results = []

        for filepath in sorted(image_files):
            print(f"\nProcessing {os.path.basename(filepath)} ...")
            image_results, exit_all, _ = process_image(filepath, annotated_dir)

            # Append whatever pairs were clicked for this image
            if image_results:
                all_results.extend(image_results)

            # If user hit ESC, break out of the loop immediately
            if exit_all:
                print("Exit requested. Saving what we have so far...")
                break

        # Save to Excel (even if no results, we create an empty file with headers)
        output_path = os.path.join(folder, "contrast_results.xlsx")
        df = pd.DataFrame(all_results, columns=[
            "filename", "x1", "y1", "x2", "y2",
            "I_max", "I_min", "ΔI", "Michaelson"
        ])
        df.to_excel(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Annotated images saved in: {annotated_dir}")



    def compute_snr_with_roi(self, flag=0):
        from tkinter import Tk, filedialog
        from tkinter.messagebox import showinfo
        showinfo("SNR", "Running SNR (with ROI) computation...")

        Tk().withdraw()
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder or not os.path.exists(folder):
            print("Invalid folder path.")
            return

        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(image_extensions)
        ]
        if not image_files:
            print("No images found.")
            return

        annotated_dir = os.path.join(folder, "annotated-snr-roi")
        os.makedirs(annotated_dir, exist_ok=True)

        results = []
        stop_flag = False

        for filepath in sorted(image_files):
            if stop_flag:
                break

            results, stop_flag = process_image_snr_roi(filepath, annotated_dir, flag, results)

            if stop_flag:
                print("User stopped the process with ESC.")
                break

        if results:
            df = pd.DataFrame(results)
            out_excel = os.path.join(folder, "snr_results_roi.xlsx")
            df.to_excel(out_excel, index=False)
            print(f"\nResults saved to: {out_excel}")
            print(f"Annotated images saved in: {annotated_dir}")
        else:
            print("No valid SNR results to save.")



    def compute_snr_with_points(self, flag=0):
        # Replace with your actual SNR logic
        QMessageBox.information(self, "SNR", "Running SNR (with points) computation...")
        
        from tkinter import Tk, filedialog
        Tk().withdraw()
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder or not os.path.exists(folder):
            print("Invalid folder path.")
            return

        image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(image_extensions)
        ]
        if not image_files:
            print("No images found.")
            return

        annotated_dir = os.path.join(folder, "annotated-snr-points")
        os.makedirs(annotated_dir, exist_ok=True)

        results = []
        stop_flag = False

        for filepath in sorted(image_files):
            if stop_flag:
                break

            results, stop_flag = process_image_snr_points(filepath, annotated_dir, flag, results)
            if stop_flag:  # ESC was pressed inside the function
                print("User stopped the process with ESC.")
                break
        if results:
            df = pd.DataFrame(results)
            out_excel = os.path.join(folder, "snr_results_points.xlsx")
            df.to_excel(out_excel, index=False)
            print(f"\nResults saved to: {out_excel}")
            print(f"Annotated images saved in: {annotated_dir}")
        else:
            print("No valid SNR results to save.")


# Run the app
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageMetricApp()
    window.show()
    sys.exit(app.exec_())
