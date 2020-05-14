# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed.
np.random.seed(1)

# Create the dataset.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Create the Neural Network.
class NeuralNetwork(object):

    def __init__(self, inputSize, outputSize, hiddenSize, lr):
        self.W1 = np.random.uniform(size=(inputSize, hiddenSize))
        self.B1 = np.random.uniform(size=(1, hiddenSize))
        self.W2 = np.random.uniform(size=(hiddenSize, outputSize))
        self.B2 = np.random.uniform(size=(1, outputSize))
        self.learningRate = lr

    def sigmoid(self, valuesIN):
        return 1 / (1 + np.exp(-valuesIN))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def tanh(self, valuesIN):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def forward(self, inputX):
        self.Z1 = np.dot(inputX, self.W1)
        self.Z1 += self.B1
        self.Ai = self.sigmoid(self.Z1)
        
        self.Z4 = np.dot(self.Ai, self.W2)
        self.Z4 += self.B2
        y_Hat = self.sigmoid(self.Z4)
        return y_Hat 

    def backpropagation(self, X, y, o):
        self.error = y - o
        self.o_delta = self.error * self.sigmoidPrime(o)

        self.Ai_error = self.o_delta.dot(self.W2.T)
        self.Ai_delta = self.Ai_error * self.sigmoidPrime(self.Ai)

        # Updating the weights and the biases.
        self.W2 += self.Ai.T.dot(self.o_delta) * self.learningRate
        self.B2 += np.sum(self.o_delta, axis=0, keepdims=True) * self.learningRate
        self.W1 += X.T.dot(self.Ai_delta) * self.learningRate
        self.B1 += np.sum(self.Ai_delta, axis=0, keepdims=True) * self.learningRate

    def train(self, X, y):
        o = self.forward(X)
        self.backpropagation(X, y, o)
        return o

if __name__ == "__main__":

    # Instantiate the class.
    NN = NeuralNetwork(2, 1, 2, 0.1)

    # Set the number of epochs.
    epochs = 10000
    
    # Get the shape of the ouput and initilize an empty list to cath the loss.
    m = y.shape[1]
    lossVEpoch = []

    # Loop through the network.
    for i in range(epochs):
        y_Hat = NN.train(X, y, 10000)
        loss = (1 / (2 * m)) * np.sum(np.square(y - y_Hat))
        lossVEpoch.append(loss)

    # Make a prediction.
    print("Prediction: ")
    print(y_Hat)

    # Plot the models loss.
    plt.plot(lossVEpoch)
    plt.title("Loss V Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
