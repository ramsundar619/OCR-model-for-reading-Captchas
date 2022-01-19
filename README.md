# OCR-model-for-reading-Captchas
## Introduction
This example demonstrates a simple OCR model built with the Functional API. Apart from combining CNN and RNN, it also illustrates how you can instantiate a new layer and use it as an "Endpoint layer" for implementing CTC loss.
## Table of Contents
- Introduction
- Installation
- Loading Dataset
- Getting Started
- Methodology being used for OCR using Keras and Tensorflow library
- Input
- Parameters used for libraries
- Output
## Installation
**Create a conda virtual environment and activate it**
```
conda create --name tf24
conda activate tf24
```
**Install Dependencies**
```
conda install tensorflow==2.4.1=gpu_py38h8a7d6ce_0
conda install matplotlib
conda install numpy==1.19.5
```
## Loading Dataset
```
!curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -qq captcha_images_v2.zip
```
## Getting Started
**For Training**
```
python3 main.py
```
**For Testing**
```
python3 test.py
```
## Methodology being used for OCR using Keras and Tensorflow library
- tf.compat.v1.ConfigProto.gpu_options.allow_growth = True (prevents tensorflow from allocating the totality of a gpu memory)
- tf.compat.as_bytes() - Converts bytearray, bytes, or unicode python input types to bytes.
- tf.keras.preprocessing.image_dataset_from_directory() - Generates a tf.data.Dataset from image files in a directory.
  - Parameters used
  - directory = "images" (Directory where the data is located).
  - validation_split = 0.2 (Optional float between 0 and 1, fraction of data to reserve for validation).
  - subset="training" (One of "training" or "validation". Only used if validation_split is set).
  - seed=1337 (Optional random seed for shuffling and transformations).
  - image_size=image_size (Size to resize images to after they are read from disk).
  - batch_size=batch_size (Size of the batches of data).
- plt.figure() - Create a new figure.
- plt.subplot() - Add an Axes to the current figure.
- plt.imshow() - Display data as an image.
- plt.title() - Set a title for the axes.
- plt.axis() - Convenience method to get or set some axis properties.
- tf.keras.layers.experimental.preprocessing.StringLookup() - A preprocessing layer which maps string features to integer indices.
- tf.io.read_file() - Reads the contents of file
- tf.io.decode_png() - Decode and convert to grayscale
- tf.image.convert_image_dtype() - Convert to float32
- tf.image.resize() - Resize to the desired size
- tf.transpose() - Transpose the image
- tf.strings.unicode_split() - Splits each string in input into a sequence of Unicode code points
- tf.data.Dataset.from_tensor_slices() - To get the slices of an array in the form of objects
- tf.Keras.layers.Layer() - A layer is a callable object that takes as input one or more tensors and that outputs one or more tensors. 
- tf.keras.backend.ctc_batch_cost() - Runs CTC loss algorithm on each batch element.
- tf.cast() - Casts a tensor to a new type.
- tf.shape() - Returns size of tensor.
- tf.ones() - Creates a tensor with all elements set to one (1).
- tf.keras.Input() - used to instantiate a Keras tensor.
- tf.keras.layers.Conv2D() - 2D convolution layer (e.g. spatial convolution over images).
  - Parameters used
  - filters - Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
  - kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
  - strides - An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
  - padding - one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. 
  - kernel_initializer - Initializer for the kernel weights matrix.
  - name - name of the layer.
  - activation - Activation function to use
- tf.keras.layers.MaxPooling2D() - Max pooling operation for 2D spatial data.
- tf.keras.layers.Reshape() - Layer that reshapes inputs into the given shape.
- tf.keras.layers.Add() - Layer that adds a list of inputs.
- tf.keras.layers.Dropout() - Applies Dropout to the input.
- tf.keras.layers.Dense() - regular densely-connected NN layer.
- tf.keras.layers.Bidirectional() - Bidirectional wrapper for RNNs.
- tf.keras.layers.LSTM() - Long Short-Term Memory layer.
- tf.keras.activations.softmax - Softmax converts a vector of values to a probability distribution.
- tf.keras.Model.compile() - Configures the model for training.
  - Parameters used
  - Optimizer - An optimizer is one of the two arguments required for compiling a Keras model.
- tf.keras.optimizers.Adam - Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order                                    moments.
- tf.keras.model.get_layer() - Retrieves a layer based on either its name (unique) or index.
- tf.saved_model.save() - Exports a tf.Module (and subclasses) obj to SavedModel format.
- tf.keras.models.load_model() - Loads the saved model.
- tf.keras.backend.ctc_decode() - Decodes the output.
- tf.expand_dims() - Returns a tensor with a length 1 axis inserted at index axis.
## Input
Here we are using [captcha_images_v2 Dataset](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip) as an input to the model which predicts the text in a given captcha image once trained.
Another possible dataset - [MJSynth dataset](https://www.robots.ox.ac.uk/~vgg/data/text/) - 10GiB 
## Parameters used
- image_size = (200, 50) - A standard dimension we fix for all images before feeding into the model.
- batch_size = 16 - batch size for training and testing.
- downsample_factor = 4  - reduce the dimensionality to minimize the possibility of overfitting
- train_size=0.9 - fraction of data to reserve for Training.
- input_encoding="UTF-8" - Splits each string in input into a sequence of Unicode code points.
- buffer_size=tf.data.AUTOTUNE - This allows later elements to be prepared while the current element is being processed.
- filters - The dimensionality of the output space (i.e. the number of output filters in the convolution).
- kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
- strides - An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. 
- padding - one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. 
- kernel_initializer - Initializer for the kernel weights matrix
- "he_normal" - It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor
- "relu" - Rectified Linear Unit Function which returns element-wise max(x, 0).
- "softmax" - Softmax converts a vector of values to a probability distribution.The elements of the output vector are in range (0, 1) and sum to 1.
- rate - Float between 0 and 1. Fraction of the input units to drop. 
## Ouput
After training the model for 100 epochs, it scores a loss of 0.0137 and val_loss of 0.0012
