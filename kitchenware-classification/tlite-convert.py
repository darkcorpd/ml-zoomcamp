# Create tensorflow-lite instance

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('model.h5')

tf.saved_model.save(model, 'tlite_model')