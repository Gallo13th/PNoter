import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
# 位置编码
def positional_embedding(maxlen, model_size):
    PE = np.zeros((maxlen, model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_size))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
    PE = tf.constant(PE, dtype=tf.float32)
    return PE

def create_model():
    inputs = layers.Input(shape=(16,))
    # conv1 = layers.Conv1D(filters=4,kernel_size=12,strides=1)(inputs)
    # conv2 = layers.Conv1D(filters=4,kernel_size=8,strides=1)(inputs)
    # conv3 = layers.Conv1D(filters=4,kernel_size=4,strides=1)(inputs)
    conv1 = layers.Dense(32)(inputs)
    conv2 = layers.Dense(32)(inputs)
    return keras.models.Model(inputs,outputs=[conv1,conv2])

print(create_model())
plot_model(create_model(),to_file="model.png", show_shapes=True, show_layer_names=False, rankdir='TB')