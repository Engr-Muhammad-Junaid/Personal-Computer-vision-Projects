from ultralytics import YOLO
import cv2
#you can download the others version as well
model=YOLO('../Yolo-Weights/yolov8n.pt')
results=model('images/1.png',show=True)
cv2.waitKey(0)