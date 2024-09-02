import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_inputs=np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

training_outputs=np.array(
    [
        [0,1,1,0]
    ]
).T

np.random.seed(1)

synaptic_weights=2*np.random.random((3,1))-1

bias = np.random.random((1, 1))


print(f"Random starting synaptic weights:\n {synaptic_weights}")

for iteration in range(20000):

    input_layer=training_inputs

    outputs=sigmoid(np.dot(input_layer,synaptic_weights)+bias)

    error= training_outputs-outputs

    adjustments=error*sigmoid_derivative(outputs)

    synaptic_weights+=np.dot(input_layer.T, adjustments)
    bias += np.sum(adjustments)


print(f"Synaptic weights after training:\n {synaptic_weights}")

print(f"Outputs after training:\n {outputs}")