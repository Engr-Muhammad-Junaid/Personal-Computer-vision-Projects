from ultralytics import  YOLO
import cv2
import cvzone
import math
#cap=cv2.VideoCapture for webcam
cap=cv2.VideoCapture('Videos\\ppe-1.mp4')
#cap.set(3,1280)
#cap.set(4,780)
model=YOLO('ppe.pt')
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    #after getting ypur resutls for the images finding the all boxes.
    for r in results:
        boxes=r.boxes
        for box in boxes:

            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
           # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #x1, y1, w, h =box.xywh[0]

            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            # getting the class name

            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1)







    cv2.imshow('image',img)
    #give one mili second delay
    cv2.waitKey(1)
