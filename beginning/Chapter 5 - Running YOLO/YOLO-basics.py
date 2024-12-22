from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-weights/yolov8l.pt') # Load model from web if not found locally
results = model('test-images/3.png', show=True) # Run inference on image and show bounding boxes
cv2.waitKey(0) # Keep window open to see results