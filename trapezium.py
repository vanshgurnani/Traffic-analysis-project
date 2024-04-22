import cv2
import numpy as np

camera_url = 'http://192.168.23.223:8080/video'

# Set output width and height
output_width = 640
output_height = 480

# Function to calculate trapezoidal ROI coordinates
def calculate_trapezoidal_roi(image_width, image_height):
    # Define the points of the trapezoidal ROI based on the lane or camera aspect ratio
    # This is just a placeholder. Adjust these coordinates based on your requirements.
    top_left = (int(0.2 * image_width), int(0.3 * image_height))
    top_right = (int(0.8 * image_width), int(0.3 * image_height))
    bottom_left = (0, image_height)
    bottom_right = (image_width, image_height)
    return [top_left, top_right, bottom_right, bottom_left]

# Main function to capture frames from video file and process them
def main():
    # Open video capture from file
    cap = cv2.VideoCapture(camera_url)  # Replace 'video/test201.mp4' with your video file path

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Resize the frame to the specified output width and height
        frame = cv2.resize(frame, (output_width, output_height))

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Calculate trapezoidal ROI coordinates
        trapezoidal_roi = calculate_trapezoidal_roi(width, height)

        # Draw trapezoidal ROI on the frame
        cv2.polylines(frame, [np.array(trapezoidal_roi, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the resulting frame
        cv2.imshow('Video Output', frame)

        # Print the aspect ratio of the output video
        aspect_ratio = width / height

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    print("Aspect Ratio:", aspect_ratio)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
