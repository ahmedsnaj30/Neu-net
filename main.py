import numpy as np      # matrix work
import pandas as pd     # reading data

input = pd.read_csv('train.csv')

input = np.array(input)   # Treat the data as a numpy array

m, n = input.shape
np.random.shuffle(input)


# Split into dev and training sets. Dev sets will be data portion that won't be trained on to 
# cross examined with trained data.

# Dev sets
data_d = input[0:1000].T  # Transpose first 1000 examples
Y_d = data_d[0]
X_d = data_d[1:n]         # first:last row num of pixels
X_d = X_d / 255

# Training sets
data_t = input[1000:m].T # Transpose rest of examples
Y_t = data_t[0]
X_t = data_t[1:n]
X_t = X_t / 255

#print(X_t[:, 1].shape)  #[ first_row:last_row , column_0 ]

def init_vars():
    W1 = np.random.rand(10, 784) - 0.5     # Weight between -0.5 and 0.5
    b1 = np.random.rand(10, 1) - 0.5       # Bias
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5 
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

def f_prop(W1, b1, W2, b2, X):     # Forward propagation
    Z1 = W1.dot(X) + b1            # dot product
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):               # One-Hot Encoding Labels
    arr = np.zeros((Y.size, Y.max() + 1))  # Create matrix of zeroes w/ tuple of its size
    arr[np.arange(Y.size), Y] = 1        # For each row, go to column specified by label and set to 1
    arr = arr.T
    return arr

def b_prop(Z1, A1, Z2, A2, W2, X, Y): # Back propagation
    oh_Y = one_hot(Y)
    dZ2 = A2 - oh_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2, 1)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    dB1 = 1/m * np.sum(dZ1, 1)
    return dW1, dB1, dW2, dB2

def update_vars(W1, b1, W2, b2, dW2, dB2, dW1, dB1, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(dB1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(dB2, (10,1))
    return W1, b1, W2, b2

def predict(A2):
    return np.argmax(A2, 0)

def accuracy(pred, Y):
    print ("Prediction:", pred, " Actual Label:", Y)
    return np.sum(pred == Y) / Y.size


def grad_des(X, Y, alpha, loops):  # Calculating gradient descent
    W1, b1, W2, b2 = init_vars()
    for i in range(loops):
        Z1, A1, Z2, A2 = f_prop(W1, b1, W2, b2, X)
        dW1, dB1, dW2, dB2 = b_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_vars(W1, b1, W2, b2, dW2, dB2, dW1, dB1, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = predict(A2)

            print("Accuracy: ", accuracy(predictions, Y))

    return W1, b1, W2, b2

W1, b1, W2, b2 = grad_des(X_t, Y_t, 0.10, 500)