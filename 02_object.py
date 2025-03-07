
import math

import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
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

while True:
    ret, frame = cap.read()
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            # for opencv
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 234), 3)

            # for cvzone
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # confidence
            # conf = math.ceil((box.conf[0] * 100))
            # # print(conf)
            # # cvzone.putTextRect(
            # #     frame,
            # #     f"{conf}",
            # #     (max(0, x1), max(35, y1)),
            # #     scale=2,
            # #     thickness=3,
            # #     offset=5,
            # # )
            # # # class name
            conf = int(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(
                frame,
                f"{className[cls]}{conf}",
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
            )
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
