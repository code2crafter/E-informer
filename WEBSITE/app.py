from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import random
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = r'WEBSITE\static\uploads'
DETECTED_FOLDER = r'WEBSITE\static\detected_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# Placeholder model paths (update with your actual trained weights)
ARDUINO_MODEL_PATH = "runs/detect/AU_trained/weights/best.pt"
RASPBERRY_MODEL_PATH = r"runs/detect/PI_trained/weights/best.pt"
CUSTOM_MODEL_PATH = r"runs/detect/component_trained/weights/best.pt"

# Load models (replace paths above if needed)
arduino_model = YOLO(ARDUINO_MODEL_PATH)
raspberry_model = YOLO(RASPBERRY_MODEL_PATH)
custom_model = YOLO(CUSTOM_MODEL_PATH)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/choose_board')
def choose_board():
    return render_template('choose_board.html')

# @app.route('/upload_arduino')
# def upload_arduino():
#     return render_template('upload_arduino.html')

# @app.route('/upload_raspberry')
# def upload_raspberry():
#     return render_template('upload_raspberry.html')

# @app.route('/upload_custom')
# def upload_custom():
#     return render_template('upload_custom.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")

# Upload routes per board
@app.route("/upload_arduino", methods=["GET", "POST"])
def upload_arduino():
    if request.method == "POST" and 'pcb' in request.files:
        file = request.files['pcb']
        if file.filename != '':
            filename = secure_filename(file.filename)

            # Save uploaded image to uploads folder
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Run YOLO model
            results = arduino_model(upload_path)

            # Create full path for detected image using app.root_path
            output_folder = os.path.join(app.root_path, "static", "detected_images")
            os.makedirs(output_folder, exist_ok=True)  # Make sure folder exists
            output_path = os.path.join(output_folder, "arduino_output.jpg")

            # Save detected image using OpenCV
            output_img = results[0].plot()
            cv2.imwrite(output_path, output_img)

            return render_template("upload_arduino.html", show_detected=True, random=random.randint(1, 10000))

    return render_template("upload_arduino.html", show_detected=False)


@app.route("/upload_raspberry", methods=["GET", "POST"])
def upload_raspberry():
    if request.method == "POST" and 'pcb' in request.files:
        file = request.files['pcb']
        if file.filename != '':
            filename = secure_filename(file.filename)

            # Save uploaded image
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Run YOLO model for Raspberry Pi
            results = raspberry_model(upload_path)

            # Save detected image
            output_folder = os.path.join(app.root_path, "static", "detected_images")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "raspberry_output.jpg")

            output_img = results[0].plot()
            cv2.imwrite(output_path, output_img)

            return render_template("upload_raspberry.html", show_detected=True, random=random.randint(1, 10000))

    return render_template("upload_raspberry.html", show_detected=False)


@app.route("/upload_custom", methods=["GET", "POST"])
def upload_custom():
    if request.method == "POST" and 'pcb' in request.files:
        file = request.files['pcb']
        if file.filename != '':
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Run YOLO detection
            results = custom_model(upload_path)
            output_img = results[0].plot()

            # Save the detected image
            output_path = os.path.join(DETECTED_FOLDER, "custom_output.jpg")
            cv2.imwrite(output_path, output_img)

            return render_template("upload_custom.html", show_detected=True, random=random.randint(1, 10000))

    return render_template("upload_custom.html", show_detected=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
