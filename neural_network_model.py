import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases for the input and hidden layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = 0.3  # Learning rate
        self.lr_decay = 0.95  # Learning rate decay factor
        self.min_lr = 0.001

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        # Initialize the weights and biases for the hidden and output layers
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, x):
        # Forward propagation
        self.hidden_layer_input = (
            np.dot(x, self.weights_input_hidden) + self.bias_hidden
        )
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights_hidden_output)
            + self.bias_output
        )
        self.output_layer_output = sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, x, y):
        # Backpropagation with cross-entropy loss
        output = self.forward(x)
        error = y - output

        # Calculate gradients
        delta_output = error

        # Update weights and biases
        self.weights_hidden_output += (
            self.hidden_layer_output.T.dot(delta_output) * self.lr
        )
        delta_hidden = (
            delta_output.dot(self.weights_hidden_output.T)
            * self.hidden_layer_output
            * (1 - self.hidden_layer_output)
        )
        self.weights_input_hidden += x.T.reshape(-1, 1).dot(delta_hidden) * self.lr
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.lr
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs, batching=True, batch_size=16):
        show_loss_every = 100 if batching else 1
        show_f1_every = 1000 if batching else 1
        prev_loss = float("inf")  # Store previous loss to check for stagnation
        consecutive_bad_epochs = 0
        for epoch in range(epochs):

            indices = np.arange(len(X))
            np.random.shuffle(indices)

            if batching:
                batch_indices = indices[:batch_size]
            else:
                batch_indices = indices
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]

            for j in range(len(x_batch)):
                x = x_batch[j]
                target = y_batch[j]
                self.backward(x, target)

            loss = self.calculate_cross_entropy_loss(X, y)
            if epoch % show_loss_every == 0:

                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                print(f"learning rate: {self.lr}")

            if epoch % show_f1_every == 0:
                f1 = self.compute_f1_score(X, y)
                print(f"Epoch {epoch}, F1 Score: {f1:.4f}")

            # Decay the learning rate
            if loss > prev_loss:
                consecutive_bad_epochs += 1
                if consecutive_bad_epochs >= 3:
                    self.lr *= self.lr_decay  # Decrease the learning rate
            else:
                consecutive_bad_epochs = 0

            prev_loss = loss

            # Ensure the learning rate doesn't go below a minimum value
            if self.lr < self.min_lr:
                self.lr = self.min_lr

    def predict(self, x):
        return np.round(self.forward(x))

    def calculate_cross_entropy_loss(self, X, y):
        predictions = self.forward(X)
        # Avoid division by zero and numerical instability
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return np.mean(loss)

    def compute_f1_score(self, X, y):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        predictions = []

        for i in range(len(X)):
            x = X[i]
            target = y[i]
            prediction = self.predict(x)
            predictions.append(prediction)  # Store predictions

            if target == 1 and prediction == 1:
                true_positives += 1
            elif target == 0 and prediction == 1:
                false_positives += 1
            elif target == 1 and prediction == 0:
                false_negatives += 1
        if true_positives == 0 and (false_positives == 0 or false_negatives == 0):
            return 0
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        # Calculate F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return f1
