import cv2
import numpy as np

# Load video file
cap = cv2.VideoCapture('video/test401.mp4')

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Background subtraction
    fg_mask = bg_subtractor.apply(blur_frame)

    # Thresholding
    _, thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

    # Contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bounding box detection
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust threshold based on your requirements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
