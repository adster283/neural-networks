# Setup development environment to run the python scripts

Open a terminal in the the top directory of this project and start a venv with the following commands:

`python -m venv myenv`

`source myvenv/bin/activate`

`pip install -r requirements.txt`

You can now follow the instructions for running the scripts.

# Single layer perceptron

I based the code on a tutorial from GeeksForGeeks with some changes to better fit my needs.
https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/

## Running the single layer perceptron script

To run the script you can use the following command:

`python3 perceptron.py ionosphere.data`

The above command assumes you are the the current directory and are wanting to train on the ionosphere data. You can easily change the data file for another by passing the path to the desired data.

*The script is configured to train for atleast 100 epochs and until if has no information gain for more than 21 epochs*

# Neural Network

This model is based on skeleton code with all function definitions outlined but not implemented. I once again made use of a GeeksForGeeks blog post for understanding how to implement the code.
https://www.geeksforgeeks.org/backpropagation-in-machine-learning/

## Running the neural network

Running the code is a simple as running the a2Part1.py script inside the algorithms/NeuralNetworks directory. This script will first train on the training set, before running on the test set to compute the accuracy.
