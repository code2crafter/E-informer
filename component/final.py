from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import os

# Load the trained YOLOv8 model
model = YOLO(r"C:\AMOL\runs\detect\component_trained\weights\best.pt")

# Open a file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if image_path:
    print(f"Selected file: {image_path}")

    # Run inference on the selected image
    results = model.predict(source=image_path, save=True, conf=0.25)

    # Access and display the first result
    result = results[0]
    result.show()
else:
    print("No image selected.")
