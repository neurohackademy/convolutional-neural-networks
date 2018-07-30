---
title: "Keras"
teaching: 5
exercises: 5
questions:
- "How do we classify images with a neural network?"
- "How can we build neural networks with Keras?"
- "How are models fit and evaluated in Keras?"
objectives:
- "Implement a neural network that classifies clothing categories from images"
keypoints:
- "Building neural networks with Keras is straightforward"
- "We need to use training, validation, testing datasets to avoid over-fitting"
---

### Image classification

Consider the task of classifying images of clothing. We have three classes
of images: Dresses, t-shirts an shoes

We read in this data, which has conveniently already been split into
training and testing sets using numpy:

~~~
import numpy as np
fashion = np.load('fashion.npz')

train_data = fashion['train_data']
train_labels = fashion['train_labels']

test_data = fashion['test_data']
test_labels = fashion['test_labels']
~~~
{: .python}

Let's look at this data:

~~~
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
~~~
{: .python}


~~~
((600, 28, 28, 1), (600, 3), (100, 28, 28, 1), (100, 3))
~~~
{: .python}

There are a total of 700 images, each with 28 by 28 pixels and one
channel (because the images are black and white). There are labels in
one-hot encoded format to go along with these images. The total sample is
already split into a training set and a test set, for the purpose of
independent cross-validation.

> #### One-hot encoding
>
> This is a mathematically convenient way to represent class information
> Each row represents one sample, and each column represents one of the
> classes (for example, "t-shirt"). The one-hot encoded array is equal to one
> in each row for the column corresponding to the class of this sample.
> For example, an array with the following entries:
>
>   `np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]`
>
> represents 4 items: a t-shirt, a shoe, a dress and another dress.
>
>

Let's visualize a few of the samples:

~~~
for im in range(10):
    fig, ax = plt.subplots(1)
    ax.set_title(train_labels[im])
    ax.imshow(train_data[im].squeeze())
~~~
{: .python}

One way to use Keras to classify these samples is by building a neural
network that we will call a "fully-connected" network. This network is
made out of layers where each layer is connected to all of the units in
the previous layer (similar to what we saw before). For example, each
unit in the first layer of the network will be connected to each one of
the pixels in the image through a weight. Subsequent layers would be
connected through a weight to each one of the units in this layer, and so
on.

We import two items from Keras:

~~~
from keras.models import Sequential
from keras.layers import Dense
~~~
{: .python}

The first is the model object. This will serve as a container to hold all
of the layers in the model. It's a `Sequential` model, because information
flows through the network in a sequential manner, from one layer to the next.

The `Dense` layer implements the fully connected idea: these are layers that
are connected to each one of the items in their inputs.

These kinds of layers do not know how to deal with mutli-dimensional
data, such as images and the spatial structure of the image doesn't
matter anyway because each one of the connections is initialized
independently and randomly, and learning proceeds from there. So we need to
reshape each image to a one-dimensional array:

~~~
train_data_1d = train_data.reshape((train_data.shape[0], -1))
test_data_1d = test_data.reshape((test_data.shape[0], -1))
~~~

The number of rows in these arrays is still the number of samples in each
portion of the data, and the number of columns is the total number of
pixels in each image: 784.

This is then the shape that the `Dense` input layer would then expect for
each sample.


Let's construct a two-layer network first:

~~~
model = Sequential()
model.add(Dense(10, input_shape=(train_data.shape[-1],), activation='relu'))
model.add(Dense(3, activation='softmax'))
~~~
{: .python}

The first layer of this network is densely connected to the image: this
is why we need to provide to it the input shape (you can verify that this
is `(784, )`). We tell this layer to use a non-linear activation
function. This is the `f()` that is applied to the computed activations
of each unit. Here, we use the rectified linear unit that we saw before,
shortened as 'relu'.

The second layer is the output layer of this model. It has the same
number of units as there are classes. In this case, three. Because it
receives its input from another layer in the network, we don't need to
specify the `input_shape` for this layer. The activation function
of this layer is a softmax function. This function is defined as:

$$f(y_i) = \frac{e^{y_i}}{\sum_i{e^{y_i}}}$$

and assigns each of the possible outputs a probability (a number between
0-1, all sum together to 1).

It also has the effect of 'mutual inhibition' between alternative options.

To be able to work faster, Keras needs to know what all the computations
are before it can start doing them. This is not our usual imperative
paradigm of programming, where we tell the computer what to do and how to
do it. Instead, Keras allows us to specify what needs to be done and lets
the computer sort out the details. Because of this, one more step is
required when constructing a model in Keras, which is a compilation step:

At the minimum, we need to specify the loss function that we would want
to use -- categorical cross-entropy is appropriate for classification --
and the optimizer that we would like to use. Here, we use the Adam
optimizer. We can alternatively also specify additional metrics that we
would like the model to report as it learns.

~~~
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
~~~

After the model is compiled, it can be fit to data:

~~~
model.fit(train_data, train_labels, epochs=10, batch_size=10, validation_split=0.2)
~~~
{: .python}

We specify the batch size, and the number of epochs that we would use.
During each epoch the model sees 10 samples at a time, and the samples
are fed through the network, until the network produces a prediction for
these samples, for each sample, we can then back-propagate an error
through the network as we saw above. In each batch this is essentially
averaged, and a step is taken in the direction of the gradient. The Adam
optimizer uses an adaptive strategy to set the learning rate at each
learning step and each parameter, making it very effective at learning
the parameters.

At the end of each epoch, we will check whether the loss has changed and
whether it's headed in the right direction, but to avoid over-fitting to
the training data, we do so on a separate validation set. We set aside
20% of the training data specifically just for this purpose. Setting
`validation_split` to 0.2 implements this.

After we are done training, we can evaluate the model on the test data
that we have set aside for this purpose:

~~~
model.evaluate(test_data, test_labels, batch_size=10)
~~~
{: .python}
`
This tells us both the loss and the metric that we asked for -- accuracy,
which is calculated as the average of the matches between the label
associate with the sample, and the class of the model prediction for this
sample.

### Making a deeper model

To make a deeper model, we can add layers. For example:

~~~
model = Sequential()
model.add(Dense(10, input_shape=(train_data.shape[-1],), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

~~~
{: .python}

Here, we've simply added one more dense layer with 10 units. The model is
compiled and fit just as before. Given the amount of data we are using
here, this does not result in a substantial change in performance, but if
we had a lot data this may have made some difference.
