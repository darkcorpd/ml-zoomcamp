# Create tensorflow-lite instance
# Imports

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient

import tensorflow.lite as tflite

# Load saved model
# Change the name of the model

model = keras.models.load_model('model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

