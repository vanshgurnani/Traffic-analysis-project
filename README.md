# Traffic Density Monitoring using YOLO and OpenCV

## Introduction

This Python script utilizes the YOLO (You Only Look Once) object detection model and OpenCV to monitor traffic density in a given video. The script identifies vehicles within a specified region of interest, calculates their density, and dynamically adjusts a traffic light simulation based on the density threshold.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`): You can install it using `pip install opencv-python`
- YOLO pre-trained weights, configuration file, and class names file. These files should be downloaded and placed in the same directory as the script. You can find them on the [official YOLO website](https://pjreddie.com/darknet/yolo/).

## Usage

1. **Install Dependencies:**

    ```bash
    pip install opencv-python
    ```

2. **Download YOLO Files:**

    Download the YOLO pre-trained weights (`yolov3.weights`), configuration file (`yolov3.cfg`), and class names file (`coco.names`). Place these files in the same directory as the script.

3. **Run the Script:**

    ```bash
    python yolo_video1.py
    ```

4. **Adjust Parameters (Optional):**

    - You can modify the script parameters such as video file path, output dimensions, lane dimensions, density threshold, and skip frames to suit your specific scenario.

## Features

- Real-time monitoring of traffic density in a specified region of interest.
- Dynamic adjustment of a traffic light simulation based on the density threshold.
- Object detection for vehicles excluding persons, traffic lights, parking meters, and stop signs.

## Acknowledgments

- YOLO for providing a robust object detection model.
- OpenCV for its versatile computer vision library.
