#!/usr/bin/env python
# coding: utf-8

import os
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor

classes = [
    'cup',
    'fork',
    'glass',
    'knife',
    'plate',
    'spoon'
]

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'tlite_model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_4'].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_3'].float_val
    result = dict(zip(preds, classes))
    max_prob = result[max(result)]
    return max_prob

def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/darkcorpd/ml-zoomcamp/main/kitchenware-classification/cup.jpg'
    response = predict(url)
    print(response)
    #app.run(debug=True, host='0.0.0.0', port=9696)
