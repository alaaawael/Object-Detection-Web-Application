# Object-Detection-Web-Application
YOLOv3 Object Detection Web App

A Flask-based web application for object detection using the YOLOv3 model, pre-trained on the COCO dataset. Upload images to detect objects, view results with bounding boxes, and save processed images.
Table of Contents

Features

Prerequisites
Installation
Usage
Configuration
Notes
Troubleshooting
Contributing
License

Features

Upload images (PNG, JPG, JPEG, JFIF) for object detection.
Uses YOLOv3 to detect 80 object classes from the COCO dataset.
Displays bounding boxes with class labels and confidence scores.
Supports images up to 16MB.
Saves processed images in an output directory.


Prerequisites

Python 3.7 or higher
TensorFlow (1.x or 2.x with compatibility mode)
Flask, Pillow (PIL), NumPy, OpenCV (cv2), Seaborn
Web browser for accessing the app
yolov3.weights (download from YOLO official site)
coco.names (file with COCO class names)

Installation

Clone the repository:
git clone https://github.com/alaaawael/your-repo.git



Set up a virtual environment):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Download YOLOv3 weights:Download yolov3.weights from the official YOLO website and place it in the project directory. 
Verify directory structure:
your-repo/
├── app.py
├── object_detection.py
├── yolov3.weights
├── coco.names
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── result.html
├── uploads/  
├── output/   



Usage

Run the Flask application:
python app.py

The app starts at http://localhost:5000.

Access the web interface:Open http://localhost:5000 in a browser.

Upload an image:

Select an image (PNG, JPG, JPEG, or JFIF).
Click "Upload" to process the image with YOLOv3.
View the result with bounding boxes and labels.


View results:Processed images are saved in the output directory with the prefix detected_.


Configuration
In app.py:

UPLOAD_FOLDER: uploads (for uploaded images).
OUTPUT_FOLDER: output (for processed images).
ALLOWED_EXTENSIONS: png, jpg, jpeg, jfif.
MAX_CONTENT_LENGTH: 16MB (max upload size).
YOLOv3 parameters:
model_size: (416, 416)
max_output_size: 10
iou_threshold: 0.5
confidence_threshold: 0.5



Notes

TensorFlow runs in compatibility mode for 1.x and 2.x support.
YOLOv3 model is initialized once at app startup for efficiency.
yolov3.weights must be in the project directory.
Debug mode is enabled by default (debug=True). Disable in production.
Font fallback to default if arial.ttf is unavailable.

Troubleshooting

File not found: Ensure yolov3.weights and coco.names are in the project directory.
TensorFlow issues: Check TensorFlow compatibility with your Python version.
Image not processed: Verify image size (<16MB) and format.
Font errors: The app uses default font if arial.ttf is missing.




License
This project is licensed under the MIT License.


