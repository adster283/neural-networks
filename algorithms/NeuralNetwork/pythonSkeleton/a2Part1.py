import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import os

from NeuralNetwork import Neural_Network


def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


if __name__ == '__main__':
    # Get the current dir path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct a path to the training data file
    train_csv = os.path.join(current_dir, 'penguins307-train.csv')
    # Load in the training data file
    data = pd.read_csv(train_csv)
    # the class label is last!
    print("Example of data row in df: ", data.head(5))
    labels = data.iloc[:, -1]
    # seperate the data from the labels
    instances = data.iloc[:, :-1]
    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])

    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))

    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

    # Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    nn.train([instances[0]], integer_encoded, 1)


    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # Train for 100 epochs, on all instances.
    nn.train(instances, integer_encoded, 100)
    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # Run the test data set to see accuracy
    # Construct path to the test data file
    test_csv = os.path.join(current_dir, 'penguins307-test.csv')
    pd_data_ts = pd.read_csv(test_csv)
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]
    #scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    print("Testing...")
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(test_labels)
    predictions = nn.predict(test_instances)
    acc = 0
    for i in range(len(predictions)):
        if predictions[i] == integer_encoded[i]:
            acc += 1
    print('Test accuracy = ', acc / len(predictions) * 100, "%")
