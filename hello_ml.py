import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# Outputs we expect 
expected_output = np.array([[0],[1],[1],[0]])

# Initialize weights
inputLayer_neurons = 2 
hiddenLayer_neurons = 2
outputLayer_neurons = 1

weights_input_hidden = np.random.uniform(size=(inputLayer_neurons,hiddenLayer_neurons))
weights_hidden_output = np.random.uniform(size=(hiddenLayer_neurons,outputLayer_neurons))

# Training algorithm
for _ in range(20000):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs,weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output,weights_hidden_output)
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * 0.1
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * 0.1

print(predicted_output)
