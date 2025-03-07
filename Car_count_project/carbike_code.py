import math
import cv2
import cvzone
import numpy as np
from sort import *
from ultralytics import YOLO

# Load video
cap = cv2.VideoCapture("car-video.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video frame size: {width} x {height}")

# Load YOLO model
model = YOLO("../yolo_weights/yolov8n.pt")

# Class names
className = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "rickshaw",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suit", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load and resize mask image
mask = cv2.imread('D:\Computer-Vision\object-detection\Car_count_project\mask .png')
if mask is None:
    print("Error: Could not load mask image.")
    exit()
mask = cv2.resize(mask, (width, height))

# Initialize tracker
tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

# Apply limit line
limits = [0, 161, 639, 170]
total_counts = []  # Store unique vehicle IDs

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    imgRegion = cv2.bitwise_and(frame, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            current_class = className[cls]

            # Detect Cars, Trucks, Buses, Motorcycles, and Bicycles
            if current_class in ["car", "truck", "bus", "motorcycle", "bicycle"] and conf > 0.3:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    # Update tracker
    resultTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (234, 0, 78), 4)

    # Draw tracking boxes and count vehicles
    for result in resultTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cvzone.cornerRect(frame, (x1, y1, w, h), l=6, t=2, colorR=(255, 75, 98))
        cvzone.putTextRect(frame, f"{id}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=4)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        # Check if vehicle crosses the counting line
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in total_counts:
                total_counts.append(id)

    # Display count
    cvzone.putTextRect(frame, f"Count: {len(total_counts)}", (50, 50), scale=1, thickness=2, offset=4)

    # Show video frames
    cv2.imshow("frame", frame)
    cv2.imshow("mask", imgRegion)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
