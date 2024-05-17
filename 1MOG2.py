import cv2

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Menggunakan webcam (0)
camera = cv2.VideoCapture("cctv.mp4")

while True:
    ret, frame = camera.read()

    # Every frame is used both for calculating the foreground mask and for updating the background. 
    foreground_mask = MOG2_subtractor.apply(frame)

    # Threshold if it is bigger than 240 pixel is equal to 255 if smaller pixel is equal to 0
    # create binary image, it contains only white and black pixels
    ret, threshold = cv2.threshold(foreground_mask.copy(), 120, 255, cv2.THRESH_BINARY)
    
    # Dilation expands or thickens regions of interest in an image.
    dilated = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
    
    # Find contours 
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check every contour if it exceeds certain value, draw bounding boxes
    for contour in contours:
        # If area exceeds certain value then draw bounding boxes
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("Subtractor", foreground_mask)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(30) & 0xff == 27:
        break
        
camera.release()
cv2.destroyAllWindows()
