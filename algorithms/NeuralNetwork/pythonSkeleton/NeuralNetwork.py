import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1.0 / (1.0 + np.exp(-input)) 
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(len(inputs)):
                weighted_sum += inputs[j] * self.hidden_layer_weights[j][i]
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            weighted_sum = 0.
            # Access weights for output node i directly
            for j in range(len(hidden_layer_outputs)):
                weighted_sum += hidden_layer_outputs[j] * self.output_layer_weights[j][i]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        hidden_layer_outputs = np.array(hidden_layer_outputs)
        output_layer_outputs = np.array(output_layer_outputs)

        output_layer_betas = np.zeros(self.num_outputs)
        # Encode 0 as [1, 0, 0], 1 as [0, 1, 0], and 2 as [0, 0, 1] (to fit with our network outputs!)
        if desired_outputs == [0]:
            output_errors = [1, 0, 0] - output_layer_outputs
        elif desired_outputs == [1]:
            output_errors = [0, 1, 0] - output_layer_outputs
        elif desired_outputs == [2]:
            output_errors = [0, 0, 1] - output_layer_outputs

        output_layer_betas = output_errors * (1 - output_layer_outputs) * output_layer_outputs
        print('OL betas: ', output_layer_betas)


        hidden_layer_betas = np.zeros(self.num_hidden)
        hidden_errors = np.dot(self.output_layer_weights, output_layer_betas)
        hidden_layer_betas = hidden_layer_outputs * (1 - hidden_layer_outputs) * hidden_errors
        print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))

        # Calculate output layer weight changes.
        for i in range(len(hidden_layer_outputs)):
            for j in range(len(output_layer_betas)):
                delta_output_layer_weights[i][j] = hidden_layer_outputs[i] * output_layer_betas[j]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(len(inputs)):
            for j in range(len(hidden_layer_betas)):
                delta_hidden_layer_weights[i][j] = inputs[i] * hidden_layer_betas[j]
        # Calculate hidden layer weight changes.
        

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # Update the weights.
        self.output_layer_weights += self.learning_rate * delta_output_layer_weights
        self.hidden_layer_weights += self.learning_rate * delta_hidden_layer_weights

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = self.predict([instance])
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)


            # Print accuracy achieved over this epoch
            acc = 0
            for i in range(len(predictions)):
                if predictions[i] == desired_outputs[i]:
                    acc += 1
            print('acc = ', acc / len(predictions))

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)

            rounded_outputs = np.round(output_layer_outputs)
            #if rounded_outputs == [1, 0, 0]:
            if np.array_equal(rounded_outputs, [1, 0, 0]):
                predicted_class = 0
            #elif rounded_outputs == [0, 1, 0]:
            elif np.array_equal(rounded_outputs, [0, 1, 0]):
                predicted_class = 1
            elif np.array_equal(rounded_outputs, [0, 0, 1]):
            #elif rounded_outputs == [0, 0, 1]:
                predicted_class = 2
            else:
                # This only happens if model is unable to make a prediction
                predicted_class = None
            predictions.append(predicted_class)
        return predictions