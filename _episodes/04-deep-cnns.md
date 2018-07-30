---
title: "Deep CNNs"
teaching: 15
exercises: 0
questions:
- "How can we make CNNs more efficient?"
- "What can we do to intepret CNN results?"

objectives:
- "Implement networks with pooling"
- "Implement networks with dropout"

keypoints:
- "Some additional tricks can improve your networks"
- "Visualizing what your network does can help interpret the results"
---

### Reducing the number of parameters: pooling

We saw that deep CNNs can have a lot of parameters. One way to reduce the number
of parameters is to condense the output of the convolutional layers, and summarize it.

A simple way to do that is to pool the pixel intensities in the output
for small spatial regions. This is similar to the convolution operation
that we saw, but instead of multiplying a small window with a kernel and
summing it, we now do a maximum operation over this window, and replace a
group of pixels with the brightest one of them.

Let's see what that would be like using numpy. We start by allocating the
output. Notice that we've managed to reduce the number of pixels by a
factor of 4. Then, we go over each pixel in the output and for each one
we choose the maximum of the corresponding little two-by-two window in
the input.

~~~
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])
~~~
{: .python}


In Keras, this operation is implemented as a layer. Here, we add a
max-pooling operation after each convolutional layer. The input to the
`MaxPool2D` units is the size of the pooling window (here 2-by-2):

~~~
from layers import MaxPool2D
cnn = Sequential()
cnn.add(Conv2D(10, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))
cnn.add(MaxPool2D(2))
cnn.add(Conv2D(10, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(2))
cnn.add(Flatten())
cnn.add(Dense(3, activation='softmax'))
~~~
{: .python}

Let's see how this affects the number of parameters:

~~~
cnn.summary()
~~~

That's pretty dramatic!

### Other strategies: regularization through dropout

Another approach to using the data more efficiently is regularization.
One approach to regularizing CNNs is called dropout. Here, instead of
fitting all of the weights each time, we ignore some randomly selected
units in each step of learning. This means that different parts of the
data (and hence different samples from the noise) affect different units.
This also allows different units to develop their own 'identity'.

Implementing this in Keras is again a matter of adding another layer. We
add a dropout layer after the convolutional layer for which we would like
to apply the dropout. The input to the `Dropout` objects is the
proportion of the units that we would like to ignore in each step of learning:

~~~
from layers import Dropout
cnn = Sequential()
cnn.add(Conv2D(10, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(10, kernel_size=3, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(3, activation='softmax'))
~~~
{: .python}


### Tracking learning

During learning, the weights used by the network change, and as they
change, the network becomes more attuned to the features of the images
that discriminate between classes. This means that the loss function we
use for training becomes smaller and smaller. Looking at the change in
the loss with learning can be helpful to see whether learning is
progressing as expected, and whether the network has learned enough.

Here's how you might track learning:

~~~
learning = model.fit(train_data, train_labels, epochs=10, batch_size=10, validation_split=0.2)

import matplotlib.pyplot as plt
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
~~~

This shows you the loss for both training and validation data. Initially, both of these will go down, as the network starts learning the features that help it classify. But eventually, the network will start overfitting. Training loss will continue to go down, but from that point on the validation loss will plateau, or even start to go back up. Knowing at which point this happens is useful.

### Storing weights before overfitting

Knowing that the model might overfit, we might want some way to recover
the weights from the best possible validation point. This can be done
using a `ModelCheckpoint` object:
~~~
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss',
                             save_best_only=True)
callbacks_list = [checkpoint]
model.fit(train_data, train_labels, validation_split=0.2, epochs=3,
          callbacks=callbacks_list)
~~~
{: .python}

Loading back these weights is done through the model's `load_weights` method:

~~~
model.load_weights('weights.hdf5')
~~~
{: .python}

Once the weights are loaded, you can keep training (maybe you got more
data?) or use the weights for inference.

### Interpreting the results

One of the main criticisms of convolutional neural networks is that they
are "black boxes" and that even when they work very well, it is hard to
understand why they work so well. Many efforts are being made to improve
the interpretability of neural networks, and this field is likely to
evolve rapidly in the next few years. One of the major thrusts of this
evolution is that people are interested in visualizing what different
parts of the network are doing. Here, I will show you how to take apart a
trained convolutional network, select particular parts of the network and
analyze their behavior.

Here, we'll look at one simple way of visualizing the model weights:
through the output of the convolutional kernels for one particular input.

The following code pulls out the weights of the first kernel in the first
layer of a trained network.
~~~
conv1 = model.layers[0]
weights1 = conv1.get_weights()
kernels1 = weights1[0]
kernel1_1 = kernels1[:, :, 0, 0]
~~~
{: .python}

Let's look at what this kernel does to one of the images in our test set

~~~
test_image = test_data[3, :, :, 0]
~~~
{: .python}

~~~
filtered_image = convolution(test_image, kernel1_1)
plt.matshow(filtered_image)
~~~
{: .python}

This is one way to interpret the results, but there are several other
ways that people have used. For example, you might start with white
noise, and then use gradient descent to shape the noise, so that it
optimally "stimulates" a particular unit. This is a way to find out what
kinds of things this unit "likes".

If you think this kind of thing is interesting, you should check out the
[recent paper by Chris Olah and colleagues](https://distill.pub/2017/feature-visualization/).
