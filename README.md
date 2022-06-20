# Neural Network

Created a neural network using the numpy library, pandas library, and MNIST database to classify images of hand-written digits. This program uses data from 28x28 pixel training images (784 nodes) from kaggle. Each pixel has a value associated with it between 0-255, indicating the lightness or darkness of the pixel. The dataset consists of columns starting with the label of the actual number written by the user, followed by the pixel values associated with the hand-written image. This neural network will read data, perform processing on the data using activation functions, and make a prediction on what number was written. With each iteration, the program is trained by reducing deviations from the labels, improving the accuracy of the predictions.
https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv


# Math
## Forward Propagation: 
Running image through network, perform activation function processing, and get a prediction of what number the image represents
X = Pixels
Y = Labels
T = Transpose 
m = rows, n = columns

A[0] = X (784 pixels * m) --> input layer; 

Z[1] = W[1]X + b[1] --> Unactivated first layer: 
Applying a weight - a matrix that obtains dot product between that matrix and A[0] input matrix. Dot product is the product of 2 vectors' magnitudes and cosine of the angle between them.
```A * B = |A||B|cosθ```
We're multiplying a bunch of weights that correspond to each of the 7840 connections between 1st/2nd layers of NN. Then add a constant bias to each node

A[1] = gReLU(Z[1])) --> apply ReLU activation function, similar to tanh/sigmoid.
This will add a complexity to the linear combination from the input + 1st layer.
One of the conditions for the universal approximation theorem to be valid is that
the neural network is a composition of nonlinear activation functions

Z[2] = W[2]A[1] + b[2] --> Unactivated second layer

A[2] = softmax(Z[2]) --> Apply softmax function to obtain probabilities of output layer
```σ(Z) = e^Zi / Σ e^Zj```

## Backwards propagation:
Reverse the propagation process to see how much our predictions deviate from actual label before next iteration
dZ[2] = A[2] − Y --> Calculate the error of second layer
Predictions - actual labels; One-Hot encode correct labels into array
https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f 

dW[2] = (1/m)dZ[2] * A[1]T --> Figure how much the weight contributed to error.
Obtain derivative of the loss function with respect to weights in the second
layer.

dB[2] = (1/m)ΣdZ[2] --> Figure how much the bias contributed to error.
Find the average of the absolute error of the second layer

dZ[1] = W[2]T * dZ[2].∗g[1]′(Z[1]) --> Figure out how much the hidden layer was off by.
Reversing the propagation proccess; Taking the errors from the second layer and
applying the weights in reverse to get the first layer's errors. You also undo the
activation by getting its derivative, g′.

dW[1] = (1/m)dZ[1] * A[0]T --> Figure how much the weight contributed to error in 0th layer

dB[1] = (1/m)ΣdZ[1] --> Figure how much the bias contributed to error in 0th layer


## Variable Updates: Reduces the error calculated from backwards propagation for next iteration
W[2]:=W[2]−αdW[2]
 
b[2]:=b[2]−αdb[2]
 
W[1]:=W[1]−αdW[1]

b[1]:=b[1]−αdb[1]
