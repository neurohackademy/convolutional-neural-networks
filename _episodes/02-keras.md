---
title: "Keras"
teaching: 5
exercises: 5
questions:
- ""
objectives:
- ""
keypoints:
- ""
- ""
---

###

Consider the task of classifying images of clothing. We have three classes
of images: Dresses, t-shirts an shoes

We read in this data, which has conveniently already been split into
training and testing sets using numpy:

~~~
import numpy as np
fashion = np.load('fashion.npz')

train_data = fashion['train_data']
test_data = fashion['test_data']
~~~
{: .python}