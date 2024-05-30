from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the model
model = load_model(r'C:\Users\DELL\Downloads\gp_model (1)\gp_model\final-modelcnn.keras')

# Preprocess the image
IMAGE_SIZE = 224
def preprocess_image(img):
    img = image.load_img(img, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

classnames=['Apple___Apple_scab', 'Apple___Black_rot', 
            'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
            'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    # Check if the file is a valid image
    if file and allowed_file(file.filename):
        img = 'temp.jpg'  # Save the image temporarily

        # Save the image to a temporary location
        file.save(img)

        # Preprocess the image
        img_array = preprocess_image(img)

        # Pass the image through the model
        predictions = model.predict(img_array)

        # Get the predicted label
        predicted_label = np.argmax(predictions, axis=1)[0]

        # Get class indices
        class_indices = model.predict(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))).argmax(axis=-1)

        return jsonify({'predicted_label': int(predicted_label), 'class_indices': classnames[predicted_label]})

    return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run()