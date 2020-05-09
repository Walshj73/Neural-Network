# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Create the input matrix.
X = np.array(([0.5726, 0.5833]), dtype = float)
y = np.array(([0.7500]), dtype = float)

# Create a class - Neural Network.
class Neural_Network(object):

    # Constructor to set up the weights of the network.
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        self.learningRate = 0.5

    # Sigmoid function.
    def sigmoid(self, valuesIN):
        return 1 / (1 + np.exp(-valuesIN))

    # Forward pass through the network.
    def forward(self, inputX):
        self.Zi = np.dot(inputX, self.W1)
        self.Ai = self.sigmoid(self.Zi)
        self.Z4 = np.dot(self.Ai, self.W2)
        y_Hat = self.sigmoid(self.Z4)
        return y_Hat

    # Derivative of Sigmoid.
    def sigmoidPrime(self, s):
      return s * (1 - s)

    # Backpropagate through the network.
    def backpropagation(self, X, y, o):
        self.o_error = o - y
        self.o_delta = np.array(([self.o_error * self.sigmoidPrime(o)]), dtype = float)

        self.Ai_error = self.o_delta.dot(self.W2.T)
        self.Ai_delta = self.Ai_error * self.sigmoidPrime(self.Ai)

        self.W1 -= self.learningRate * np.array(([X]), dtype = float).T.dot(self.Ai_delta)
        self.W2 -= self.learningRate * np.array(([self.Ai]), dtype = float).T.dot(self.o_delta)

    # Train the Network.
    def train(self, X, y):
      o = self.forward(X)
      self.backpropagation(X, y, o)

lossVEpoch = []

# Instantiate the class.
NN = Neural_Network()

# Loop through the network.
for i in range(100):
  print("\n*********************\nEpoch", (i+1), "\n*********************\n")
  y_Hat = NN.forward(X)
  print(y_Hat[0])

  # Print the predicted score & the loss.
  print("Predicted Score: ", round(y_Hat[0] * 100, 2), "%")
  loss = 0.5 * (np.square(y - NN.forward(X)))
  print("Loss: ", loss)
  NN.train(X, y)
  lossVEpoch.append(loss)

# Plot the models loss.
plt.plot(lossVEpoch)
plt.title("Loss V Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
