import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO



model = YOLO("yolo11n.pt")
train_results = model. train(
data=r"C:\AMOL\arduino_uno\dataset\data.yaml",
epochs=100,
imgsz=640,
workers=0,
batch=16,
name="AU_trained",

)