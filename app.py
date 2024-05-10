#import subprocess
#import sys

#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#packages = ["opencv-python", "kaggle", "transformers", "ultralytics", "torch", "pandas", "numpy", "tensorflow", "requests"]

#for package in packages:
 #   install(package)
#

from flask import Flask, request, render_template, jsonify,render_template_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ultralytics import YOLO
import os
import torch
import re

app = Flask(__name__)

# Load YOLO model
model_path = os.getcwd()+'/train21/weights/best.pt'  # Change this to your model's path
model_best_LVIS  = YOLO(model_path)

def format_label(label):
    # Regular expression to find numbers and items
    match = re.match(r'(\d+)\s(.+)', label)
    if not match:
        return "Invalid label format"
    
    number, item = match.groups()
    number = int(number)  # Convert to integer to handle numbers correctly

    if number == 1:
        return f" {number} {item.rstrip('s').replace('_', ' ')}"
    else:
        return f"{number} {item.replace('_', ' ')}"




# Suppress all other warnings
#warnings.filterwarnings('ignore')

# Load GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_description(image_path):
    """Generate a description from the image using YOLO and GPT-2."""
    # Predict using YOLO
    prediction_results_LVIS = model_best_LVIS.predict(source=image_path)

    labels = "describe the image with"+ prediction_results_LVIS[0].verbose()

    prompt = format_label(labels.strip(','))

    # Generate text with GPT-2
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    output_sequences = gpt_model.generate(
        input_ids=encoded_input,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=0.7,
        do_sample=True,
        top_k=75,
        early_stopping=True
    )
    description = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return description

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            description = generate_description(file_path)  # Make sure this function is defined
            return render_template_string(HTML_TEMPLATE, result=description)
    return render_template_string(HTML_TEMPLATE)
    
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Upload an Image</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
    <div class="container">
        <h1>Upload an image and get a description</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" id="file" class="upload-box">
            <button type="submit" class="btn">Upload</button>
        </form>
        {% if result %}
            <div class="result">
                <h2>Description:</h2>
                <p>{{ result }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
