import cv2
import time
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture("walking.mp4")
while True:
    r, frame = cap.read()
    if r:
        start_time = time.time()
        frame = cv2.resize(frame, (640, 360)) # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)
        
        # Measure elapsed time for detections
        end_time = time.time()
        print("Elapsed time:", end_time-start_time)
        
        mask = np.zeros_like(gray_frame)  # Create a black mask
        
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Create a mask for the area inside the rectangle
            roi = mask[y:y+h, x:x+w]
            roi[:] = 255  # Set the region inside the rectangle to white
        
        # Apply the mask to the frame to remove the background inside the detected rectangles
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break
