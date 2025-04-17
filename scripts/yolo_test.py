from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("/home/scenescribe/Desktop/scenescribe/models/yolo.pt")
print("Model Loaded")
# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("/home/scenescribe/Desktop/image_frames/image_57.png", imgsz=640)
print(results)