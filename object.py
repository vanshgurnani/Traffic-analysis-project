import cv2
import time

# Function to detect vehicles using background subtraction and contour detection
def detect_vehicles(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Apply morphological operations to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected vehicles
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust the threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Open video capture
cap = cv2.VideoCapture('video/test201.mp4')  # Change the argument to the file path if you want to use a video file

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was properly captured
    if not ret:
        break

    # Detect vehicles and draw bounding boxes
    frame_with_boxes = detect_vehicles(frame)

    # Display the output
    cv2.imshow('Vehicle Detection', frame_with_boxes)

    # Introduce a delay between frames to regulate the processing speed (adjust the value as needed)
    time.sleep(0.03)  # Delay of approximately 33 milliseconds (for ~30 FPS)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
