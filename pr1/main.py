import math
import random


class UniversalNeuron:
    def __init__(self, input_size, lr=0.1, use_scaling=False):
        self.input_size = input_size
        self.lr = lr
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.use_scaling = use_scaling
        self.E_prev = float('inf')

    def prepare_time_series(self, data):
        dataset = []
        for i in range(len(data) - self.input_size):
            inputs = data[i: i + self.input_size]
            target = data[i + self.input_size]
            dataset.append((inputs, target))
        return dataset

    def step_activation(self, inputs, threshold):
        S = sum(x * w for x, w in zip(self.weights, inputs))
        return 1 if S >= threshold else 0

    def activate(self, inputs):
        """Time series activation."""
        S = sum(x * w for x, w in zip(self.weights, inputs))
        multiplier = 10 if self.use_scaling else 1
        return (1 / (1 + math.exp(-S))) * multiplier, S

    def train_on_batch(self, dataset):
        """Time series training."""
        total_error = 0
        accumulated_dws = [0.0] * self.input_size

        for inputs, target in dataset:
            Y_pred, S = self.activate(inputs)
            total_error += (target - Y_pred) ** 2

            sigmoid_val = 1 / (1 + math.exp(-S))
            sigmoid_der_val = sigmoid_val * (1 - sigmoid_val)

            for j in range(self.input_size):
                E_der = (Y_pred - target) * sigmoid_der_val * inputs[j]
                accumulated_dws[j] += E_der

        for j in range(self.input_size):
            self.weights[j] -= self.lr * (accumulated_dws[j] / len(dataset))

        return total_error


def run_logic():
    print("=" * 10 + " LOGIC " + "=" * 10)

    and_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    nn_and = UniversalNeuron(2)
    nn_and.weights = [1.0, 1.0]
    for (inputs, output) in and_data:
        print(
            f"\nAND \ninput: {inputs}, \noutput: {output}, \npredicted: {nn_and.step_activation(inputs, 1.5)}\n")

    print("-" * 30)

    or_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    nn_or = UniversalNeuron(2)
    nn_or.weights = [1.0, 1.0]
    for (inputs, output) in or_data:
        print(
            f"\nOR \ninput: {inputs}, \noutput: {output}, \npredicted: {nn_or.step_activation(inputs, 0.5)}\n")

    print("-" * 30)

    not_data = [([0], 1), ([1], 0)]
    nn_not = UniversalNeuron(1)
    nn_not.weights = [-1.5]
    for (inputs, output) in not_data:
        print(
            f"\nNOT \ninput: {inputs}, \noutput: {output}, \npredicted: {nn_not.step_activation(inputs, -1)}\n")


def run_xor():
    print("=" * 10 + " XOR " + "=" * 10)

    def xor_network(x1, x2):
        n1 = UniversalNeuron(2)
        n1.weights = [1.0, -1.0]
        y1 = n1.step_activation([x1, x2], 0.5)

        n2 = UniversalNeuron(2)
        n2.weights = [-1.0, 1.0]
        y2 = n2.step_activation([x1, x2], 0.5)

        n3 = UniversalNeuron(2)
        n3.weights = [1.0, 1.0]
        return n3.step_activation([y1, y2], 0.5)

    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    for (inputs, output) in xor_data:
        print(
            f"\nXOR \ninput: {inputs}, \noutput: {output}, \npredicted: {xor_network(inputs[0], inputs[1])}\n")


def run_series():
    print("\n" + "=" * 10 + " TIME SERIES " + "=" * 10)
    full_data = [1.92, 4.01, 1.48, 5.45, 1.56, 5.42, 1.28, 4.34, 1.51, 5.49, 1.32, 4.00, 0.49, 4.19, 1.53]

    nn = UniversalNeuron(input_size=3, lr=0.1, use_scaling=True)

    train_data = nn.prepare_time_series(full_data)
    test_data = [train_data.pop(), train_data.pop()]

    epoch = 0
    prev_error = float('inf')
    while True:
        curr_error = nn.train_on_batch(train_data)
        epoch += 1
        if stop_training(curr_error, prev_error, epoch):
            print(f"\nTraining stopped!\n")
            break
        prev_error = curr_error

    print(f"\nFinal weights: {[round(w, 4) for w in nn.weights]}")

    for test in test_data[::-1]:
        pred, _ = nn.activate(test[0])
        print(f"Input {test[0]} -> output: {test[1]}, predicted: {pred:.4f}")


def stop_training(current_error, prev_error, epoch):
    diff = abs(current_error - prev_error)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: E = {current_error:.6f}, Difference E0-E = {diff:.6f}")

    if diff < 0.0001:
        print(f"Epoch {epoch}: E = {current_error:.6f}, Difference E0-E = {diff:.6f}")
        return True

    if epoch > 100_000:
        print(f"Epoch {epoch}: Training stopped!")
        return True
    return False


if __name__ == "__main__":
    run_logic()
    print("\n")

    run_xor()
    print("\n")

    run_series()
    print("\n")
