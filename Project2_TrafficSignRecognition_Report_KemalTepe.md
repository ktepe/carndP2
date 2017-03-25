# Traffic Sign Recognition 

## Kemal Tepe, ketepe@gmail.common

### Objective: To build a traffic sign recognition architecture using convolutional neural networks (CNN) which can achieve 93% or more accuracy with given German road sign dataset.


### Summary:

The starting point for this project was to use Lenet-5 CNN architecture what we have learned in the Udacity Self Driving Nano-Degree (carND). Original architecture was to recognize hand written numbers. However, here we need to recognize road signs. There are more signs, 43 signs, compared to number 0-9. The rest of the document explains steps taken to complete the task. The architecture provided in this folder can recognize signs with 94-96% accucary, which exceeds the required 93%.

### Lay out of the code:

0. Set up the important libraries and imports for rest of the program such as tensorflow, pandas, numpy.
1. Load the data set 
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Summarize the results with a written report

Now we will go to individual steps of the code.

#### 0. Set up the libraries

```python
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
```

#### 1. Load the data set

The code comments provides details about the set and how they are loaded.

```python
#Reading the training, validation and testing data 
#stored in the subfolder "traffic-signs-data" folder
training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

```
#### 2. Exploratory visualization of the dataset and identify

The following code obtains necessary information such as size of the data set, shape to be used in the previous sections. The labels are also correctly identified.

```pyhton
#Number of training examples
n_train = len(X_train)

#Number of testing examples.
n_test = len(X_test)

#the shape of an traffic sign image
image_shape = X_train[0].shape

#How many unique classes/labels there are in the dataset.
n_classes = len(train)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

```

The output for this part was: 

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 4
```

#### 3. Design and Test a Model Architecture

##### ..1. Architecture

ArtThe Lenet architecture is used in the project. Before setting up the architecture, I have reviewed few key papers in the area and the following two were the most useful: [1](./sermanet-ijcnn-11.pdf) by Sermanet et al. and [2](.\lenet_chalmers) by Credi. Credi in [2] has also used Lenet to comprate his own architecture for road sign clasification. The crucial point in the Lenet for this tasks was to identify parameters such as number of hidden nodes in the layers and convolutional filter dimensions. After extensive trials of these parameters, the following architecture was constructed. 

|Table 1: Architecture | | |
|---------|--------|--------|
|Layer | Description | Parameters |
|Layer 1| CNN 5x5x1 | input=32x32x1 output=28x28x48|
|Pooling | 2x2x1 Max | input=28x28x1 output=14x14x1 |
| Activation| ELU| |
|Layer 2| CNN 5x5x1 | input=14x14x48 output=10x10x96|
|Pooling | 2x2x1 Max | input=10x10x1 output=5x5x1 |
| Activation| ELU| |
|Layer 3| fully connected | input=5x5x96 output=140|
| Activation| ELU| |
|Layer 4| fully connected | input=140 output=96|
| Activation| ELU| |
|Layer 5| fully connected | input=96 output=43|
|logits| output=43| |

The full Lenet code is given below:

```python
#my convolutional NN architecture. 
#it is derived from Lenet 5 architecture, 
#but optimized by using number of hidden nodes for the traffic sign clasification 

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.03 # hyperparameter for noise
    #set up the number of nodes between the layers
    w1nodes=48 # number of conv hidden nodes in the conv NN in the layer 1
    w2nodes=96 # number of conv hidden nodes in the conv NN in the layer 2 
    w3nodes=5*5*w2nodes #number of fully connected nodes in layer 3 and flattening 
    w4nodes=140 #number of fully connected nodes in layer 4
    w5nodes=96 #number of fully connected nodes in Layer 5
    outputnodes=43 #number of output nodes, since there are 43 signs in the data set.
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x(w1nodes).
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, w1nodes), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(w1nodes))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    #instead of RELU, i used ELU non-linear function
    conv1 = tf.nn.elu(conv1)

    # SOLUTION: Pooling. Input = 28x28x(w1nodes). Output = 14x14x(w1nodes).
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x(w2nodes).
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, w1nodes, w2nodes), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(w2nodes))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.elu(conv2)

    # SOLUTION: Pooling. Input = 10x10x1(w2nodes) Output = 5x5x(w2nodes).
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x(w2nodes). Output = w3nodes.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = w3nodes Output = w4nodes
    
    fc1_W = tf.Variable(tf.truncated_normal(shape=(w3nodes, w4nodes), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(w4nodes))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.elu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = w4nodes Output = w5nodes

    fc2_W  = tf.Variable(tf.truncated_normal(shape=(w4nodes, w5nodes), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(w5nodes))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.elu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = w5nodes Output = outputnodes.
    
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(w5nodes, outputnodes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(outputnodes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


```

##### ..2. Training, validating and testing

After the architecture set up. The training and evaluation modules are set up as follows:

```python
# setup the training parameters
#original rate = 0.001 
rate=0.001
EPOCHS = 20
BATCH_SIZE = 128
print('learning rate', rate, 'Epoch', EPOCHS, 'Batch Size', BATCH_SIZE)

# setup tensors to be feed at each epoch
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y,43)

# Train your model here.
#set up the system
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#Evaluate how well the loss and accuracy of the model for a given dataset.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

```


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ...

Here is an example of an original image and an augmented image:

alt text

The difference between the original data set and the augmented data set is the following ...

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

Layer	Description
Input	32x32x3 RGB image
Convolution 3x3	1x1 stride, same padding, outputs 32x32x64
RELU	
Max pooling	2x2 stride, outputs 16x16x64
Convolution 3x3	etc.
Fully connected	etc.
Softmax	etc.
#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an ....

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

training set accuracy of ?
validation set accuracy of ?
test set accuracy of ?
If an iterative approach was chosen:

What was the first architecture that was tried and why was it chosen?
What were some problems with the initial architecture?
How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Which parameters were tuned? How were they adjusted and why?
What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
If a well known architecture was chosen:

What architecture was chosen?
Why did you believe it would be relevant to the traffic sign application?
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

alt text alt text alt text alt text alt text

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

Image	Prediction
Stop Sign	Stop sign
U-turn	U-turn
Yield	Yield
100 km/h	Bumpy Road
Slippery Road	Slippery Road
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Probability	Prediction
.60	Stop sign
.20	U-turn
.05	Yield
.04	Bumpy Road
.01	Slippery Road
For the second image ...