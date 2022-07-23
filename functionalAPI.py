import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# map data to training + validation
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# flatten + fit vals
# 784 b/c image size 28x28 px
train_x = train_x.reshape(-1, 784).astype('float32') / 255.0  # normalize to 0-1
test_x = test_x.reshape(-1, 784).astype('float32') / 255.0  # normalize to 0-1

# create a model NN
# Functional API - more flexible, multiple i/o
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# compile and configure training part of NN
model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),  # loss calc (False b/c softmax)
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # learning rate
    metrics=['accuracy'],
)

# fit model
model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=2)

# validate model
model.evaluate(test_x, test_y, batch_size=32, verbose=2)