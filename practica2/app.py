from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

class_names = {
    0: "Personas",
    1: "Carros",
    2: "Crosswalks",

}

model = load_model('C:/Users/emili/Downloads/practica2/cnn_model.keras', compile=False)
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = image.resize((720, 480))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names.get(predicted_class_index, "Desconocido")

        return jsonify({'prediction': predicted_class_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
