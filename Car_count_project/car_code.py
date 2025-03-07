import math

import cv2
import cvzone
from sort import *
from ultralytics import YOLO

cap = cv2.VideoCapture("car-video.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Get frame width and height using CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video frame size: {int(width)} x {int(height)}")

model = YOLO("../yolo_weights/yolo11n.pt")
className = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "rickshaw",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suit",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
# Load and resize the mask image to match the video frame size
mask = cv2.imread("..\external_files\mask .png")

if mask is None:
    print("Error: Could not load mask image.")
    exit()
mask = cv2.resize(mask, (int(width), int(height)))

# Tracking
tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

# apply limit line
limits = [0, 161, 639, 170]
total_counts = []
while True:
    ret, frame = cap.read()
    imgRegion = cv2.bitwise_and(frame, mask)

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            # for opencv
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # for cvzone
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(frame, (x1, y1, w, h), l=6, t=2)
            # confidence
            conf = int(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            current_class = className[cls]

            if (
                current_class == "car"
                or current_class == "truck"
                or current_class == "bus"
                or current_class == "motorbike"
                or current_class == "bicycle"
                and conf > 0.3
            ):
                # cvzone.putTextRect(
                #     frame,
                #     f"{className[cls]}{conf}",
                #     (max(0, x1), max(35, y1)),
                #     scale=0.6,
                #     thickness=1,
                #     offset=4,
                # )
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=6, t=2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (234, 0, 78), 4)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=6, t=2, colorR=(255, 75, 98))
        cvzone.putTextRect(
            frame,
            f"{int(id)}",
            (max(0, x1), max(35, y1)),
            scale=0.6,
            thickness=1,
            offset=4,
        )
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in total_counts:

                total_counts.append(id)
            
    cvzone.putTextRect(frame, f" Count: {len(total_counts)}", (50, 50))

    cv2.imshow("frame", frame)
    cv2.imshow("mask", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
