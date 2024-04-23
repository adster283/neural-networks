import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.03):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    
    # Define the first linear layer 
    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + + self.weights[0]
        return Z
    

    def step(self, z):
        if z <= 0:
            return 1
        else:
            return 0
        

    def predict(self, inputs):
        z = self.linear(inputs)
        try:
            pred = []
            for i in z:
                pred.append(self.step(z))
        except:
            return self.step(z)
        return pred
    

    def loss(self, prediction, target):
        loss = prediction - target
        return loss
    

    def train(self, inputs, targets):
        prediction = self.predict(inputs)
        error = self.loss(prediction, targets)
        self.weights[0]  += self.learning_rate * error
        self.weights[1:] = np.add(self.weights[1:], (self.learning_rate * error * inputs), out=self.weights[1:], casting="unsafe")

    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for inputs, target in zip(X, y):
                self.train(inputs, target)

            pred_list = []
            for row in test_X:
                pred_list.append(perceptron.predict(row))
            accuracy = 0

            for i in range(len(test_y)):
                if pred_list[i] == test_y[i]:
                    accuracy += 1

            if accuracy / len(test_y) == 1:
                print("Done training!. Starting the tests...")
                break
            else:
                pass
                print("Current accuracy: ", epoch,  accuracy / len(test_y))
                

if __name__ == "__main__":


    # Reading in the data from the file
    file = "algorithms/Perceptron/ionosphere.data"
    df = pd.read_csv(file, sep=' ')

    # First normalise the data before we split it
    for column in df.columns[:-1]:
        df[column] -= df[column].mean()

    # Split the data into training and testing sets
    train = df.sample(frac=0.8)
    test = df.drop(train.index)
    train = train.to_numpy()
    test = test.to_numpy()

    # Grab the class labels
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_y = np.where(train_y == 'g', 1, 0)
    test_y = np.where(test_y == 'g', 1, 0)


    np.random.seed(23)

    perceptron = Perceptron(len(train_X[1]))

    perceptron.fit(train_X, train_y, num_epochs=100)

    pred_list = []
    for row in test_X:
        pred_list.append(perceptron.predict(row))

    accuracy = 0
    for i in range(len(test_y)):
        if pred_list[i] == test_y[i]:
            accuracy += 1
    print("Final accuracy: ", accuracy / len(test_y))


