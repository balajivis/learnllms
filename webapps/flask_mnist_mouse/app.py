from flask import Flask, request, render_template
from PIL import Image
import base64
from io import BytesIO
import os


import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model
model = load_model('mnist_cnn.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = file.filename
        file_full_path = os.path.join('./uploads', filename)
        file.save(file_full_path)
        print('File uploaded successfully')

        # Open image file
       
        img = Image.open(file_full_path).convert('L')
        img = img.resize((28, 28), Image.ANTIALIAS)
        img_data = np.array(img)

        # Normalize the data
        img_data = img_data / 255.0
        img_data = img_data.reshape(1, 28, 28, 1)

        # Use the model to predict
        prediction = model.predict(img_data)

        predicted_digit = np.argmax(prediction)
        print(predicted_digit)
        return str(predicted_digit)
    
@app.route('/predict', methods=['POST'])
def predict():  
    if request.method == 'POST':
        print(request.form)  # Debug output

        image_data = request.form['imageData']
        # Decode base64 image string
        image_bytes = base64.b64decode(image_data.split(',')[1])
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert('L')
        # Resize image to 28x28
        image = image.resize((28, 28))
        # Convert image to numpy array
        image_array = np.array(image) / 255.0
        # Reshape the image array for model input
        input_data = image_array.reshape(1, 28, 28, 1)
        # Perform the prediction using the trained model
        prediction = model.predict(input_data)
        predicted_digit = np.argmax(prediction)
        print(predicted_digit)
        return str(predicted_digit)

if __name__ == "__main__":
    app.run(port=8080)