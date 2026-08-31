from ultralytics import YOLO
import cv2

model = YOLO('yolo_weights/bt.pt')
results = model(r"F:\WORK\yolo dataset\Tumor.v1i.yolov8\valid\images\brisc2025_test_00531_me_sa_t1_jpg.rf.366edabe7a58026560f82c8126927f5b.jpg"
                , show=False)

# Get plotted image
img = results[0].plot()

cv2.imshow("Result", img)
cv2.waitKey(0)
