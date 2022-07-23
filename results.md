# CONTROL
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
None

Process finished with exit code 0


# Optimizer final accuracy (with lr = 0.001 and no other params)
* Adam: 0.9724
* SGD: 0.9009
* RMSprop: 0.9773
* Adadelta: 0.8281
* Adagrad: 0.9227
*Adamax: 0.9806
* Nadam: 0.9765
* Ftrl: 0.1135


# Activation function mod (5 Epochs, one Layer of 512 --> output layer of 10)
(512 relu, 10) - 0.9809
(512 softmax, 10) - 0.8710
(512 sigmoid, 10) - 0.9780
(512 softplus, 10) - 0.9805
(512 softsign, 10) - 0.9765
(512 tanh, 10) - 0.9753
(512 selu, 10) - 0.9753
(512 elu, 10) - 0.9770
(512 exponential, 10) - 0.9732
