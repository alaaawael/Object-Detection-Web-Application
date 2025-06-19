import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from object_detection import Yolo_v3, load_class_names, draw_boxes, load_images, load_weights

# Initialize Flask app
app = Flask(__name__)

# Generate a secure secret key (in production, use environment variables)
app.secret_key = secrets.token_hex(32)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize YOLO model (only once when app starts)
class_names = load_class_names('coco.names')
model = Yolo_v3(
    n_classes=len(class_names),
    model_size=(416, 416),
    max_output_size=10,
    iou_threshold=0.5,
    confidence_threshold=0.5
)

# TensorFlow setup
tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, [1, 416, 416, 3])
detections = model(inputs, training=False)
model_vars = tf.compat.v1.global_variables(scope='yolo_v3_model')
assign_ops = load_weights(model_vars, 'yolov3.weights')

# Create session and load weights
sess = tf.compat.v1.Session()
sess.run(assign_ops)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', classes=class_names)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Allowed file types are: ' + ', '.join(app.config['ALLOWED_EXTENSIONS']))
        return redirect(request.url)
    
    try:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Process the image
        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Run detection
        img_array = load_images([input_path], model_size=(416, 416))
        detection_result = sess.run(detections, feed_dict={inputs: img_array})
        
        # Draw boxes and save result
        draw_boxes([input_path], detection_result, class_names, (416, 416))
        
        # Rename output to our desired name
        temp_output = os.path.join('output', os.path.basename(input_path))
        if os.path.exists(temp_output):
            os.rename(temp_output, output_path)
            return redirect(url_for('show_result', filename=output_filename))
        else:
            flash('Error: Could not find processed image')
            return redirect(request.url)
            
    except Exception as e:
        flash(f'Error during processing: {str(e)}')
        return redirect(request.url)

@app.route('/output/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)

@app.route('/output_image/<filename>')
def output_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File is too large (max 16MB)')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)