import cv2
from ultralytics import YOLO

import cvzone
import time
import math
import numpy

model = YOLO('yolo_weights/yolov8n.pt')

cap = cv2.VideoCapture(r"video/video1.mp4")
cap.set(3,1080)
cap.set(4,1920)

past_frame_time = time.time()

while True :
    new_frame_time = time.time()
    success , img = cap.read()
    if not success: #safety check
        break
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h = y2-y1
            w = x2-x1
            cvzone.cornerRect(img,(x1,y1,w,h))
            score = round(float(box.conf[0]),2)
            cls = model.names[int(box.cls[0])]
            cvzone.putTextRect(img, f"{cls} {score}" , (max(1,x1), max(40,y1)) ,scale=2  , thickness = 1)
    diff = new_frame_time - past_frame_time

    if diff > 0:
        fps = 1 / diff
    else:
        fps = 0

    past_frame_time = new_frame_time
    print(fps)
    cv2.imshow("Output", img)
    cv2.waitKey(1)


