import sys
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
        
        # Variable to count the epochs since we hit a highest accuracy
        epochs_no_improvement = 0
        highest_accruacry = 0
        epoch_count = 0

        #for epoch in range(num_epochs):
        while True:
            for inputs, target in zip(X, y):
                self.train(inputs, target)

            pred_list = []
            for row in test_X:
                pred_list.append(perceptron.predict(row))
            accuracy = 0

            for i in range(len(test_y)):
                if pred_list[i] == test_y[i]:
                    accuracy += 1

            num_accuracy = accuracy / len(test_y)

            

            if num_accuracy == 1:
                print("Reached 100% accuracy, stopping training")
                break
            else:
                pass

                print("Current accuracy: ", num_epochs,  num_accuracy)

                if num_accuracy > highest_accruacry:
                    highest_accruacry = num_accuracy
                    epochs_no_improvement = 0
                elif epochs_no_improvement >= 21:
                    print("No improvement for more than 21 epochs. Ending the training...")
                    if num_epochs >= 100:
                        break
                    else:
                        epochs_no_improvement += 1
                        epoch_count += 1
                else:
                    epochs_no_improvement += 1
                    epoch_count += 1
                

if __name__ == "__main__":


    # Get the file to load from sys args
    if len(sys.argv) != 2:
        print("Please path to the data file...")

    # Reading in the data from the file
    try:
        #file = "algorithms/Perceptron/ionosphere.data"
        file = sys.argv[1]
        df = pd.read_csv(file, sep=' ')
    except:
        print("Unable to load the data file. Please ensure you are passing in the absolute file path...")

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
    print("Final accuracy: ", accuracy / len(test_y) * 100,  "%")
    print("The weights used for final test were:", perceptron.weights)


