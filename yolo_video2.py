import cv2
import numpy as np
import time
import tkinter as tk
from threading import *
from queue import Queue
from PIL import Image, ImageTk
import multiprocessing

# Define a global variable for the queue
queue = Queue()

def video_process(path, processed_frame_label):
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    # Open video capture
    cap = cv2.VideoCapture(path)

    # Define desired output width and height
    output_width = 854
    output_height = 480

    density_threshold = 40

    # Define lane dimensions
    lane_width = 1000
    lane_center_x = output_width // 2
    lane_center_y = output_height

    # Flag to indicate if density threshold is reached
    density_threshold_reached = False

    # Timer variables
    start_time = None
    end_time = None

    # Calculate shape dimensions
    shape_top_width = lane_width * 0.57
    shape_bottom_width = lane_width * 0.5
    shape_height = 400

    # Calculate shape coordinates
    shape_top_left_x = lane_center_x - shape_top_width // 2
    shape_top_right_x = lane_center_x + shape_top_width // 2
    shape_bottom_left_x = lane_center_x - shape_bottom_width // 2
    shape_bottom_right_x = lane_center_x + shape_bottom_width // 2
    shape_top_y = lane_center_y - shape_height
    shape_bottom_y = lane_center_y

    # Set the desired skip_frames value
    skip_frames = 15

    # Counter to keep track of frames
    frame_counter = 0

    green_light_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_counter % skip_frames != 0:
            frame_counter += 1
            continue
        # Reset the frame counter after skipping frames
        frame_counter = 1

        # Resize frame to desired output size
        frame = cv2.resize(frame, (output_width, output_height))

        height, width, _ = frame.shape

        # Draw the trapezoidal red boundary on the frame
        area_coordinates_pixel = [(shape_top_left_x, shape_top_y),
                                (shape_top_right_x, shape_top_y),
                                (shape_bottom_right_x, shape_bottom_y),
                                (shape_bottom_left_x, shape_bottom_y)]
        cv2.polylines(frame, [np.array(area_coordinates_pixel, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw the grid
        grid_color = (255, 0, 0)

        for i in range(1, 10):
            y = int((i / 10) * height)
            cv2.line(frame, (0, y), (width, y), grid_color, 1)
            cv2.putText(frame, f"{i}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for i in range(1, 10):
            x = int((i / 10) * width)
            cv2.line(frame, (x, 0), (x, height), grid_color, 1)
            cv2.putText(frame, f"{i}", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
                    if classes[class_id] in ['person', 'traffic light','parking meter','stop sign']:
                        continue
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

                for i in indices:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the area of the red box using the formula for a convex quadrilateral
                x1, y1 = area_coordinates_pixel[0]
                x2, y2 = area_coordinates_pixel[1]
                x3, y3 = area_coordinates_pixel[2]
                x4, y4 = area_coordinates_pixel[3]

                red_box_area = (0.0002645833) * (0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) + x4 * (y2 - y1)))

                # Calculate density percentage
                density_percentage = (num_objects_after_nms / red_box_area) * 100
                
                # algorithm for time calculation 
                # Check if density threshold is reached and start the timer
                if density_percentage >= density_threshold and not density_threshold_reached:
                    start_time = time.time()
                    density_threshold_reached = True

                # Check if density threshold is still reached
                if density_threshold_reached:
                    elapsed_time = time.time() - start_time
                    elapsed_time_text = f"Elapsed Time: {elapsed_time:.2f} seconds"
                    cv2.putText(frame, elapsed_time_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if elapsed_time <= 15:
                        print(f"Time to reach density threshold: {elapsed_time:.2f} seconds")
                        print(f"Density at that time: {density_percentage:.2f}%")
                    elif elapsed_time > 15 and green_light_time is None:
                        # If elapsed time is more than 15 seconds and green light time is not set
                        green_light_time = elapsed_time + 10  # Set green light time after additional 10 seconds
                        print("Waiting for additional 10 seconds before setting signal to green...")
                        print(green_light_time)
                        light_color = (0, 255, 0)  # Green
                        light_text = "Green Light"
                    elif green_light_time is not None and time.time() >= green_light_time:
                        # If green light time is set and current time is past green light time
                        print("Setting signal to green...")
                        # Reset variables for next cycle
                        density_threshold_reached = False
                        start_time = None
                        green_light_time = None
    
                # GUI Implementtation
                if density_percentage >= density_threshold:
                    light_color = (0, 255, 0)  # Green
                    light_text = "Green Light"
                    queue.put("green" )
                else:
                    light_color = (0, 0, 255)  # Red
                    light_text = "Red Light"
                    queue.put("red")

                # Draw the area and density text on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                area_text = f"Red Box Area: {red_box_area:.2f} sq. unit"
                density_text = f"Density: {density_percentage:.2f}%"
                light_text = f"Traffic Light: {light_text}"
                cv2.putText(frame, area_text, (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, density_text, (10, 90), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, light_text, (10, 120), font, 0.7, light_color, 2, cv2.LINE_AA)

                # Draw count text on the frame
                count_text = f"Total Vehicle Count: {num_objects_after_nms}"
                cv2.putText(frame, count_text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL Image
        pil_img = Image.fromarray(frame_rgb)
        # Convert PIL Image to Tkinter PhotoImage
        tk_img = ImageTk.PhotoImage(image=pil_img)
        # Update the Label with the new image
        processed_frame_label.config(image=tk_img)
        processed_frame_label.image = tk_img  # Keep a reference to prevent garbage collection

    # Release video capture and close
    cap.release()

def simulate_traffic_light(root):
    # Create a frame for the traffic light simulation
    traffic_frame = tk.Frame(root, width=100, height=300)
    traffic_frame.pack(side=tk.LEFT)

    canvas = tk.Canvas(traffic_frame, width=100, height=300, bg="white")
    canvas.pack()

    def update_traffic_light():
        while True:
            # Check if there's a message in the queue
            if not queue.empty():
                color = queue.get()
                canvas.delete("all")  # Clear previous light
                if color == "red":
                    canvas.create_oval(25, 25, 75, 75, fill="red")  # Red light
                    canvas.create_oval(25, 100, 75, 150, outline="gray")  # Yellow light (not lit)
                    canvas.create_oval(25, 175, 75, 225, outline="gray")  # Green light (not lit)
                elif color == "green":
                    canvas.create_oval(25, 25, 75, 75, outline="gray")  # Red light (not lit)
                    canvas.create_oval(25, 100, 75, 150, outline="gray")  # Yellow light (not lit)
                    canvas.create_oval(25, 175, 75, 225, fill="green")  # Green light
                else:
                    print("Invalid color provided.")
            root.update()

    # Start updating traffic light in a separate thread
    update_thread = Thread(target=update_traffic_light)
    update_thread.start()

def main1():
    # Create the first instance of the GUI for the first video feed
    root1 = tk.Tk()
    root1.title("Traffic Monitoring System - Video 1")

    # Create input field for the first video path
    label1 = tk.Label(root1, text="Enter Video 1 Path:")
    label1.pack()

    video_path_entry1 = tk.Entry(root1)
    video_path_entry1.pack()

    # Create a frame for the processed frame label and traffic light simulation for video 1
    frame1 = tk.Frame(root1)
    frame1.pack()

    processed_frame_label1 = tk.Label(frame1)
    processed_frame_label1.pack(side=tk.RIGHT)

    # Call function to integrate traffic light simulation for video 1
    simulate_traffic_light(frame1)

    def start_processing1():
        video_path1 = video_path_entry1.get()
        video_thread1 = Thread(target=video_process, args=(video_path1, processed_frame_label1))
        video_thread1.start()

    start_button1 = tk.Button(root1, text="Start Processing Video 1", command=start_processing1)
    start_button1.pack()

    root1.mainloop()

def main2():
    # Create the second instance of the GUI for the second video feed
    root2 = tk.Tk()
    root2.title("Traffic Monitoring System - Video 2")

    # Create input field for the second video path
    label2 = tk.Label(root2, text="Enter Video 2 Path:")
    label2.pack()

    video_path_entry2 = tk.Entry(root2)
    video_path_entry2.pack()

    # Create a frame for the processed frame label and traffic light simulation for video 2
    frame2 = tk.Frame(root2)
    frame2.pack()

    processed_frame_label2 = tk.Label(frame2)
    processed_frame_label2.pack(side=tk.RIGHT)

    # Call function to integrate traffic light simulation for video 2
    simulate_traffic_light(frame2)

    def start_processing2():
        video_path2 = video_path_entry2.get()
        video_thread2 = Thread(target=video_process, args=(video_path2, processed_frame_label2))
        video_thread2.start()

    start_button2 = tk.Button(root2, text="Start Processing Video 2", command=start_processing2)
    start_button2.pack()

    root2.mainloop()


if __name__ == "__main__":
    # Create processes for main1 and main2 functions
    process1 = multiprocessing.Process(target=main1)
    process2 = multiprocessing.Process(target=main2)

    # Start both processes
    process1.start()
    process2.start()

    # Join both processes to wait for their completion
    process1.join()
    process2.join()

