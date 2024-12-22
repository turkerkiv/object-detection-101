from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) # Open webcam that has index 0
cap.set(3, 640) # Set width of webcam - propId3 for width
cap.set(4, 480) # Set height of webcam - propId4 for height

# cap = cv2.VideoCapture("videos/people.mp4") # Open video file

model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while (True):
    success, img = cap.read() # Read frame from webcam
    img = cv2.flip(img, 1) # Flip the img frame horizontally
    results = model(img, stream=True) # Show inferences on frame. stream=True means it will optimize it for real-time
    for r in results:
        boxes = r.boxes # Get the boxes from the results
        for box in boxes: 
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0] # Get the x1, y1, x2, y2 from the box. We could get x1,y1,w,h as well
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw rectangle around the object
            w, h = x2-x1, y2-y1
            bbox = x1, y1, int(w), int(h) 
            cvzone.cornerRect(img, bbox)
            
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100 # Get the confidence of the object

            # Class
            cls = int(box.cls[0]) # Get the class of the object
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(10,x1), max(20,y1-10)), 1, 1) # Put class and conf on the object
            
            
    cv2.imshow("Image", img) # Show the img frame from webcam
    cv2.waitKey(1) # Wait for 1ms before next frame