import cv2
import cvzone
import numpy as np
from sort import *
from ultralytics import YOLO

# Load video
cap = cv2.VideoCapture("people.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video frame size: {width} x {height}")

# Load YOLO model
model = YOLO("../yolo_weights/yolov8n.pt")  # Ensure correct weights path

# Load YOLO's class names dynamically
class_names = model.names  # YOLO automatically provides class names

# Load and resize mask image
mask = cv2.imread("mask2.png")
if mask is None:
    print("Error: Could not load mask image.")
    exit()
mask = cv2.resize(mask, (width, height))

# Initialize tracker
tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

# Store unique person IDs
total_counts = []

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Apply mask to frame
    imgRegion = cv2.bitwise_and(frame, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Ensure class ID is within bounds and detect only persons
            if cls in class_names and class_names[cls] == "person" and conf > 0.5:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    # Update tracker
    resultTracker = tracker.update(detections)

    # Draw tracking boxes and count persons
    for result in resultTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2  # Center point

        # Draw bounding box
        cvzone.cornerRect(frame, (x1, y1, w, h), l=6, t=2, colorR=(255, 75, 98))
        cvzone.putTextRect(frame, f"{id}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=4)

        # Add unique person ID to the count
        if id not in total_counts:
            total_counts.append(id)

    # Display person count
    cvzone.putTextRect(frame, f"Count: {len(total_counts)}", (50, 50), scale=1, thickness=2, offset=4)

    # Show frames
    cv2.imshow("frame", frame)
    cv2.imshow("mask", imgRegion)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
