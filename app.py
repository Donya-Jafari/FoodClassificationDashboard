from flask import Flask, request, render_template
from huggingface_hub import login
from datasets import load_dataset
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os


app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model('../image_classification.keras')
login('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')
x = load_dataset('OmidAghili/Image_Classification')

def preprocess_input(sample_image):
    sample_image = tf.convert_to_tensor(sample_image, dtype=tf.float32)
    sample_image = tf.image.resize(sample_image, [224, 224])
    sample_image = sample_image[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    sample_image = sample_image - mean
    sample_image = tf.expand_dims(sample_image, axis=0)
    return sample_image

def process_prediction(input):
    prediction = model.predict(input)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['file']
    if file:
        img_path = os.path.join('static', 'uploaded_image.jpg')
        image = Image.open(io.BytesIO(file.read()))
        image.save(img_path)

        input_data = preprocess_input(np.array(image))
        result = process_prediction(input_data)
        prediction = x['train'].features['label'].names[result[0]]
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
