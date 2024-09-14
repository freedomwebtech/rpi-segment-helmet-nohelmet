import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
import cvzone

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1020, 500)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO("best.pt")

# Get YOLO class names
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.2
count = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)    

    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame, imgsz=240)

    # Create an overlay for transparent polygons
    overlay = frame.copy()

    for result in results:
        if result.masks is not None and result.boxes is not None:
           for mask, box in zip(result.masks.xy, result.boxes):
               points = np.int32([mask])
               class_id = int(box.cls[0])
               class_name = yolo_classes[class_id]
            
               # Draw polygons on the overlay
               cv2.polylines(overlay, [points], True, (0,0,255), 1)
               cv2.fillPoly(overlay, [points], (0,0,255))
            
               # Draw class name text
               x, y, w, h = box.xyxy[0].numpy()  # Get bounding box coordinates and convert to int
               x1=int(x)
               y1=int(y)
               x2=int(w)
               y2=int(h)
               cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
               text_position = (int((x1 + x2) / 2), int(y1 - 10))  # Adjust text position to be above the rectangle
               cvzone.putTextRect(overlay, f"{class_name}", text_position, 1, 1)

    # Apply transparency
    alpha = 0.5  # Adjust transparency level here (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
