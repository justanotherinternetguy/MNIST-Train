import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
MODEL SUMMARY:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 535,818
Trainable params: 535,818
Non-trainable params: 0
_________________________________________________________________
"""


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
# Sequential API in keras
# not flexible - each NN layer has one in and one out
model = keras.Sequential([
    keras.Input(shape=(28*28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)  # output layer, 10 nodes for ints 0-9
])

# print(model.summary())
# import sys
# sys.exit()


# compile and configure training part of NN
model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),  # loss calc
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # learning rate
    metrics=['accuracy'],
)

# fit model
model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=2)

# validate model
model.evaluate(test_x, test_y, batch_size=32, verbose=2)

# Conclusions:
"""
learning_rate higher = faster compilation, less accurate (and vice versa)

Optimizer final accuracy (with lr = 0.001 and no other params)
Adam: 0.9724
SGD: 0.9009
RMSprop: 0.9773
Adadelta: 0.8281
Adagrad: 0.9227
Adamax: 0.9806
Nadam: 0.9765
Ftrl: 0.1135
"""