import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Predict on an image
detections = model.predict(source="D:\GIT_HUB\fullStackDatascience\bus.jpg", show=False)

print(detections)

# Display the results correctly using Boxes API
results = detections[0]
for box in results.boxes:
    # Get class name and confidence
    cls_id = int(box.cls[0])
    label = results.names[cls_id]
    score = float(box.conf[0])
    print(f"Label: {label}, Score: {score:.2f}")


