import cv2
import numpy as np

# Load YOLO
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"

print("Loading YOLO weights from:", weights_path)
print("Loading YOLO config from:", cfg_path)

net = cv2.dnn.readNet(weights_path, cfg_path)

if net.empty():
    print("Failed to load YOLO model.")
    exit()

# Load COCO classes
classes_path = "coco.names"

print("Loading COCO classes from:", classes_path)

classes = []
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Loaded", len(classes), "classes.")

layer_names = net.getLayerNames()
output_layer_names = net.getUnconnectedOutLayersNames()

# Find the indices of the output layers in the layer_names list
output_layers = [layer_names.index(layer) for layer in output_layer_names]

# Confidence threshold for detection
confidence_threshold = 0.3

# Open video capture
cap = cv2.VideoCapture('model2test.mp4')  # Replace 'model2test.mp4' with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process outputs
    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add detection to list
                detections.append((x, y, x + w, y + h))

                # Draw rectangle around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display class label and confidence
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Apply post-processing techniques
    # You can experiment with additional techniques here, such as contour analysis or morphological operations

    # Display the processed frame
    cv2.imshow('Frame', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close
cap.release()
cv2.destroyAllWindows()
