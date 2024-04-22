import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.1):
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
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0]  += self.learning_rate * error


    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for inputs, target in zip(X, y):
                self.train(inputs, target)



if __name__ == "__main__":


    # Reading in the data from the file
    file = "algorithms/Perceptron/ionosphere.data"
    data = pd.read_csv(file, sep=' ')

    # Seperate features from class labels
    numpy_array = data.iloc[:, :-1].to_numpy()
    classes = data['class'].to_numpy()
    classes = np.where(classes == 'g', 1, 0)

    print(len(numpy_array))
    print(len(classes))

    np.random.seed(23)

    perceptron = Perceptron(len(numpy_array[1]))

    perceptron.fit(numpy_array, classes, num_epochs=7)

    pred_list = []
    for row in numpy_array:
        pred_list.append(perceptron.predict(row))


    print(pred_list)

    accuracy = 0
    for row in classes:
        if pred_list[row] == classes[row]:
            accuracy += 1
    print("accuracy: ", accuracy / len(classes))


    # Plot the dataset
    plt.scatter(numpy_array[:, 0], numpy_array[:, 1], c=pred_list)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

