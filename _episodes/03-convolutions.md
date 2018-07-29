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
result = np.zeros(image.shape)
for ii in range(image.shape[0]-1):
    for jj in range(image.shape[1]-1):
        result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)
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
image through a convolution with a kernel that is defined as having low values on the left and high values on the right:

~~~
image = train_data[3, :, :, 0]

kernel = np.array([[-1, 1], [-1, 1]])

result = np.zeros(image.shape)
for ii in range(image.shape[0]-2):
    for jj in range(image.shape[1]-2):
        result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)

plt.matshow(result)

~~~
{: .python}

As we can see, this emphasizes parts of the image that match these
kernel: have dark on their left and bright on their right, such as the
left edge of the t-shirt.