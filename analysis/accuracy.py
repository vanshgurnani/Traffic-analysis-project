import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture('video/test201.mp4')  # Replace with your video path

# Define desired output width and height
output_width = 854
output_height = 480

# Define lane dimensions
lane_width = 1000  # Adjust as needed
lane_center_x = output_width // 2
lane_center_y = output_height

# Calculate shape dimensions
shape_top_width = lane_width * 0.57  # 57% of lane width
shape_bottom_width = lane_width * 0.5  # 50% of lane width
shape_height = 400  # Adjust as needed

# Calculate shape coordinates
shape_top_left_x = lane_center_x - shape_top_width // 2
shape_top_right_x = lane_center_x + shape_top_width // 2
shape_bottom_left_x = lane_center_x - shape_bottom_width // 2
shape_bottom_right_x = lane_center_x + shape_bottom_width // 2
shape_top_y = lane_center_y - shape_height
shape_bottom_y = lane_center_y

# Traffic light control variables
green_time = 20  # Green light duration in seconds
red_time = 20  # Red light duration in seconds
last_light_change = time.time()
green_light = False
previous_density_high = False
correct_predictions = 0
total_predictions = 0

# Font settings for time display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
text_color = (255, 255, 255)
text_position = (10, 150)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to desired output size
    frame = cv2.resize(frame, (output_width, output_height))

    height, width, _ = frame.shape

    # Draw the trapezoidal red boundary on the frame
    area_coordinates_pixel = [(shape_top_left_x, shape_top_y),
                              (shape_top_right_x, shape_top_y),
                              (shape_bottom_right_x, shape_bottom_y),
                              (shape_bottom_left_x, shape_bottom_y)]
    cv2.polylines(frame, [np.array(area_coordinates_pixel, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Get information about detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Define the red area
    red_area = np.array([area_coordinates_pixel], dtype=np.int32)
    red_area = red_area.reshape((-1, 1, 2))

    # Initialize a list to store objects inside the red area
    objects_inside_red_area = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Check if bounding box intersects with the red area
                if cv2.pointPolygonTest(red_area, (center_x, center_y), False) > 0:
                    # Object is inside the red area
                    objects_inside_red_area.append(len(boxes) - 1)

    # Implementing Non-Maximum Suppression (NMS)
    if len(boxes) > 0:
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Show green boxes only for objects detected inside the red area
        if len(indices) > 0:
            indices = [i for i in indices.flatten() if i in objects_inside_red_area]

            num_objects_after_nms = len(indices)

            # Drawing green boxes around detected objects on the frame
            for i in indices:
                x, y, w, h = boxes[i]
                # Draw a green rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color: (0, 255, 0), Thickness: 2

            # Calculate the area of the red box using the formula for a convex quadrilateral
            x1, y1 = area_coordinates_pixel[0]
            x2, y2 = area_coordinates_pixel[1]
            x3, y3 = area_coordinates_pixel[2]
            x4, y4 = area_coordinates_pixel[3]

            red_box_area = (0.0002645833) * (0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) + x4 * (y2 - y1)))

            # Calculate density percentage
            density_percentage = (num_objects_after_nms / red_box_area) * 100

            density_threshold = 40

            # If density goes above 40%, turn signal green for 20 sec
            if density_percentage > density_threshold:
                if not previous_density_high:
                    last_light_change = time.time()
                    green_light = True
                    previous_density_high = True
                    correct_predictions += 1
                else:
                    total_predictions += 1
            else:
                previous_density_high = False
                total_predictions += 1

            # Check if it's time to change the light
            if green_light and time.time() - last_light_change > green_time:
                last_light_change = time.time()
                green_light = False
                print("Traffic Light Changed to Red at:", time.strftime('%H:%M:%S', time.localtime(last_light_change)))

            # Determine the traffic light color
            light_color = (0, 255, 0) if green_light else (0, 0, 255)
            light_text = "Green Light" if green_light else "Red Light"

            # Draw the area and density text on the frame
            density_text = f"Density: {density_percentage:.2f}%"
            light_text = f"Traffic Light: {light_text}"
            time_text = f"Traffic Light Changed to Green at: {time.strftime('%H:%M:%S', time.localtime(last_light_change))}"
            print("Density:", density_percentage)
            print("Accuracy:", correct_predictions / total_predictions if total_predictions > 0 else 0)

            cv2.putText(frame, density_text, (10, 90), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, light_text, (10, 120), font, font_scale, light_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, time_text, (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw count text on the frame
            count_text = f"Total Vehicle Count: {num_objects_after_nms}"
            cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed frame without saving
    cv2.imshow('Processed Frame', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close
cap.release()
cv2.destroyAllWindows()
