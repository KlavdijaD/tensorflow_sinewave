import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math

SAMPLES = 1500

np.random.seed(786)
tf.random.set_seed(786)

x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
np.random.shuffle(x_values)
y_values=np.sin(x_values)
y_values += 0.078 * np.random.randn(*y_values.shape)

TRAIN_SPLIT=int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2*SAMPLES + TRAIN_SPLIT)

x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

assert(x_train.size+x_validate.size+x_test.size) == SAMPLES


model = tf.keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

training_info = model.fit(x_train, y_train, epochs=350, batch_size=64, validation_data=(x_validate, y_validate))

mae = training_info.history['mae']
validation_mae = training_info.history['val_mae']

loss = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model=converter.convert()

open("sinewave_model.tflite", "wb").write(tflite_model)

from tensorflow.lite.python.util import convert_bytes_to_c_source
source_text, header_text = convert_bytes_to_c_source(tflite_model, "sine_model")

with open('sine_model.h', 'w') as file:
    file.write(header_text)
with open('sine_model.cpp', 'w') as file:
    file.write(source_text)
