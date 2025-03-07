import cv2
from ultralytics import YOLO

model = YOLO("../yolo_weights/yolo11n.pt")

result = model("external_files\image.png")
cv2.imshow("bus", result[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
