from flask import Flask
from flask import request
from flask import jsonify
from fastapi.encoders import jsonable_encoder

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'cup',
    'fork',
    'glass',
    'knife',
    'plate',
    'spoon'
]

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    data_info = request.get_json()
    data = jsonable_encoder(data_info)
    url = data['url']
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
