import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 1  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    # Unpack arguments
    start = 0
    end = NUM_HIDDEN*NUM_INPUT
    W1 = w[0:end]
    start = end
    end = end + NUM_HIDDEN
    b1 = w[start:end]
    start = end
    end = end + NUM_OUTPUT*NUM_HIDDEN
    W2 = w[start:end]
    start = end
    end = end + NUM_OUTPUT
    b2 = w[start:end]
    # Convert from vectors into matrices
    W1 = W1.reshape(NUM_HIDDEN, NUM_INPUT)
    W2 = W2.reshape(NUM_OUTPUT, NUM_HIDDEN)

    return W1,b1,W2,b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)).T / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    b = np.zeros((labels.size, 10))
    b[np.arange(labels.size), labels] = 1
    labels = b.T
    return images, labels

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    b1 = b1.reshape(-1, 1)
    b2 = b2.reshape(-1, 1)
    z1 = W1.dot(X) + b1
    h1 = np.maximum(z1, 0)
    z2 = (W2.dot(h1)) + b2
    z2[z2 > 705] = 705
    yhat = np.exp(z2)
    yhat = yhat / np.sum(yhat, axis=0, keepdims=True)
    cost = (-1 / X.shape[1]) * np.sum(Y * np.log(yhat))
    y = np.argmax(yhat, axis=0)
    yOneHot = np.full((y.shape[0],10), np.zeros(10))
    for i in range(0, len(y)):
        yOneHot[i][y[i]] = 1
    acc = np.sum(Y.T * yOneHot, axis=1)
    acc = np.mean(acc)

    return cost, acc, z1, h1, W1, W2, yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    cost, acc, z1, h1, W1, W2, yhat = fCE(X, Y, w)
    W1, b1, W2, b2 = unpack(w)
    diff = (yhat-Y)
    g = diff.T.dot(W2) * (z1.T > 0)
    gt = g.T
    gb1 = gt
    gb1 = np.mean(g, axis=0)
    gb2 = diff
    gb2 = np.mean(diff, axis=1)
    gw1 = gt.dot(X.T) 
    gw2 = diff.dot(h1.T)
    gb1 = gb1.reshape((gb1.shape[0],))
    gb2 = gb2.reshape((gb2.shape[0],))
    wgrad = pack(gw1, gb1, gw2, gb2)
    return wgrad

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w, hyperparameters):
    epochs = 20
    hidden = hyperparameters[0]
    learningRate = hyperparameters[1]
    batchSize = hyperparameters[2]

    idxs = np.arange(0, trainX.shape[1])
    np.random.shuffle(idxs)
    iterations = int(trainX.shape[1] / batchSize)

    for i in range(epochs):
        for j in range(iterations):
            x = trainX.T[batchSize*j : batchSize*(j+1)]
            y = trainY.T[batchSize*j : batchSize*(j+1)]
            gradient = gradCE(x.T, y.T, w)
            w = w - learningRate * gradient
            
            if iterations - j <= 20 and i == epochs - 1:
                loss = fCE(testX, testY, w)[0]
                acc =  fCE(testX, testY, w)[1]
                print("Cross-Entropy:", loss)
                print("Accuracy:", acc)
    return w, loss, acc

def findBestHyperParameters(xTrain, yTrain):
    # Setting threshold for validation vs training examples
    validationThreshold = int(trainX.shape[1] * 0.8)
    xTrainingSet = xTrain.T[0:validationThreshold].T
    xValidationSet = xTrain.T[validationThreshold:].T
    yTrainingSet = yTrain.T[0:validationThreshold].T
    yValidationSet = yTrain.T[validationThreshold:].T

    bestAcc = 0
    bestHyperparameters = None
    hyperparameters = [
        (30, 0.001, 16),(30, 0.005, 64),(30, 0.001, 128),(40, 0.001, 16),(40, 0.005, 64),
        (40, 0.001, 128),(40, 0.001, 256),(50, 0.005, 16),(50, 0.01, 64),(50, 0.001, 128)]
    
    for index, params in enumerate(hyperparameters):
        W1 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
        b1 = 0.01 * np.ones(NUM_HIDDEN)
        W2 = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
        b2 = 0.01 * np.ones(NUM_OUTPUT)
        w = pack(W1, b1, W2, b2)

        weight, loss, acc = train(xTrainingSet, yTrainingSet, xValidationSet, yValidationSet, w, params)
        print("\nParameter Set #", index + 1)
        print("Layer:", params[0], "Learning Rate:", params[1], "Batch Size:", params[2], "\n")
        if acc > bestAcc:
            bestHyperparameters = params

    print("Best parameters:", bestHyperparameters)
    return bestHyperparameters

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print("Numerical gradient:")
    print(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], 1e-10))
    print("Analytical gradient:")
    print(gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w))
    print("Discrepancy:")
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], \
                                    lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_), \
                                    w))

    # Train the network using SGD.
    train(trainX, trainY, testX, testY, w, [40, 0.01, 64])
    best = findBestHyperParameters(trainX, trainY)
    
    W1 = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)
    model, loss, accuracy = train(trainX, trainY, testX, testY, w, best)

    print("\n\nRESULTS USING BEST PARAMETERS ON TEST SET:")
    print("Cross-Entropy:", loss)
    print("Accuracy:", accuracy)
