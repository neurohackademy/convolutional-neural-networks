---
title: "Convolutional networks"
teaching: 0
exercises: 0
break: 15
---

Having seen fully-connected artificial neural networks and their
implementation, we are finally ready to talk about convolutional networks.

Convolutional networks are similar to the fully connected networks that
we have seen so far, but instead of having each unit connect to all of
its inputs through independent weights, a unit in a convolutional layer
will be connected to its inputs through a small convolutional filter of
some specified size.

### What is a convolution?

A convolution is a multiplication of each patch in the image with a set
of weights and then a summing of the product. That is, each pixel in the
result of a convolution is a product of a window in the image with a
kernel function, which is a filter with fixed weights that is the size of
the window.

Let's see an implementation of a convolution in numpy for a kernel of
size 2 by 2:

~~~
def convolution(image, kernel)
    result = np.zeros(image.shape)
    for ii in range(image.shape[0]-1):
        for jj in range(image.shape[1]-1):
            result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)
    return result
~~~
{: .python}

This image also shows what a convolution does:

![](../fig/convolution_animation.gif)

Here, the source image (at the bottom) has been zero-padded, so that the
result output has the same size as the image input.

In neural networks, the result of a convolution is sometimes called a
feature map. This is because the effect of a convolution is to emphasize
the location of a particular feature in an image.

For example, consider an image of a t-shirt. We can take and run this
image through a convolution with a kernel that is defined as having low
values on the left and high values on the right:

~~~
image = train_data[3, :, :, 0]

kernel = np.array([[-1, 1], [-1, 1]])

plt.matshow(result)

~~~
{: .python}

As we can see, this emphasizes parts of the image that match these
kernel: have dark on their left and bright on their right, such as the
left edge of the t-shirt.

### Convolutions in neural networks

In neural networks convolutions are used instead of fully connected
units. This means that a unit in a first layer might have only 9 weights
(for a 3-by-3 convolution) instead of one weight per pixel. Learning
these 9 weights is still done through back-propagation, as in the
fully-connected network

Let's see how to implement this in Keras:

~~~
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=3, input_shape=(n_rows, n_cols, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
~~~
{: .python}

The model is the same `Sequential` model as before, but instead of a
Dense layer in the first layer, we have a convolutional layer, with 10
units. We specify the kernel size we'd like to have (in this case, the
kernel will be 3-by-3 pixels). We also specify the input shape. Because
the units in this layer do care about spatial arrangement, we keep the
images intact, and we tell the units in this layer to expect inputs with
spatial dimensions. In the case of the Fashion MNIST dataset,
n_rows=n_cols=28. We also keep the channels dimension (that last 1),
because we need to tell the convolutions to expect only one channel (they
can deal with more). The units in this layer use the same `'relu'`
activation function that we used before.

The last layer in the network is still a fully connected layer, that has
3 units. This is because the output of a convolution cannot be a single number.

After the convolutional layer, and before the Dense layer, we need to
convert the feature map from the convolutions into a one-dimensional
array that the Dense layer at the top can accept as input. We add a
`Flatten` layer in-between, to serve as a connector. This layer doesn't
do anything except convert the feature maps from the convolutional layer
into one-dimensional arrays.

As before, we compile the model:

~~~
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
~~~
{: .python}

And then fit it:

~~~
model.fit(train_data, train_labels, epochs=10, batch_size=10, validation_split=0.2)
~~~
{: .python}


### How many parameters?

What are the differences between fully connected and convolutional networks?

One way to approach this is to look at the numbers of parameters in these
models.

We'll construct two models with two layers each, both will have 10 units
in each layer, but one will be a CNN, while the other is a fully
connected model:

~~~
dense = Sequential()
dense.add(Dense(10, input_shape=(784, ), activation='relu'))
dense.add(Dense(10, activation='relu'))
dense.add(Dense(3, activation='softmax'))

cnn = Sequential()
cnn.add(Conv2D(10, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))
cnn.add(Conv2D(10, kernel_size=3, activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(3, activation='softmax'))
~~~
{: .python}

One way to compare the models is to look at the number of parameters that
they have:

~~~
cnn.summary()

dense.summary()
~~~

Interestingly, despite having much fewer parameters in the initial
stages, the CNN has many many more parameters at the top. That is because
to read out all the information from the feature maps at the top, you
need one weight per pixel _in each paramter map_. One way to think about
this is that the convolutions give the network a lot of expressive power,
but you need a lot of parameters to read this out.

Next, we'll see some ways to mitigate this and deal with this explosion
of paramters.
