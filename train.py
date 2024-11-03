from ultralytics import YOLO

# Update the path to your data.yaml file
data_path = "C:/Users/visha/OneDrive/Desktop/metal detection/dataset/data.yaml"

# Initialize the model
model = YOLO("ultralytics/yolov8n.pt")

# Train the model
model.train(data=data_path, epochs=50, imgsz=640)
