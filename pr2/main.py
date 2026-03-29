import json
import math
import random
from collections import deque

TRAIN_NOISE_LEVEL = 0.1
TEST_NOISE_LEVEL = 0.2

LAYERS_CONFIG = [(), (10,), (15, 5), (18, 9, 18), (10, 10, 10, 10), (36, 36)]

ACTIVATIONS_TYPE = ["sigmoid", "relu"]


GREEN = "\033[92m"
RED = "\033[91m"
GREY = "\033[90m"
RESET = "\033[0m"


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        learning_rate=0.1,
        activation_type="sigmoid",
        weights_range=(-1, 1),
    ):
        self.layers_sizes = [input_size] + list(hidden_layers) + [output_size]
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        self.layer_outputs = []

        self.weights_range = weights_range
        self.activation_type = activation_type

        self._init_params()

    def _init_params(self):
        self._init_activation_function()
        for i in range(len(self.layers_sizes) - 1):
            rows = self.layers_sizes[i + 1]
            cols = self.layers_sizes[i]
            self.weights.append(
                [
                    [random.uniform(*self.weights_range) for _ in range(cols)]
                    for _ in range(rows)
                ]
            )
            self.biases.append([random.uniform(-0.1, 0.1) for _ in range(rows)])

    def _init_activation_function(self) -> None:
        """Returns weights for the specific activation function."""
        if self.activation_type == "sigmoid":
            self.activation_func = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
            self.lr = 0.25
            self.weights_range = (-1, 1)
        elif self.activation_type == "relu":
            self.activation_func = self._relu
            self.activation_derivative = self._relu_derivative
            self.lr = 0.1
            self.weights_range = (-0.5, 0.5)

    @staticmethod
    def _sigmoid(s):
        return 1 / (1 + math.exp(-s))

    @staticmethod
    def _relu(s):
        return max(0, s)

    @staticmethod
    def _relu_derivative(relu_val):
        return 1 if relu_val > 0 else 0

    @staticmethod
    def _sigmoid_derivative(sigmoid_val):
        return sigmoid_val * (1 - sigmoid_val)

    def _forward(self, inputs):
        """
        We calculate the weighted sum and predict the current layer output(next layer input) values.
        """
        self.layer_outputs = [inputs]
        current_input = inputs

        for i in range(len(self.weights)):
            next_layer = []
            is_last_layer = i == len(self.weights) - 1

            for row_idx in range(len(self.weights[i])):
                summation = self.biases[i][row_idx]
                for col_idx in range(len(current_input)):
                    summation += (
                        self.weights[i][row_idx][col_idx] * current_input[col_idx]
                    )

                if is_last_layer and self.activation_type == "relu":
                    activation = self._sigmoid(summation)
                else:
                    activation = self.activation_func(summation)

                next_layer.append(activation)

            self.layer_outputs.append(next_layer)
            current_input = next_layer
        return next_layer

    def predict_step(self, inputs):
        output = self._forward(inputs)
        return tuple(round(o) for o in output)

    def backpropagate(self, targets, output):
        deltas: deque = self._calculate_output_deltas(targets, output)
        self._append_hidden_deltas(deltas)
        return deltas

    def _calculate_output_deltas(self, targets, output) -> deque:
        """Calculate the 'first' delta between the output and the hidden layer."""
        deltas = deque()
        deltas_layer = []
        for i in range(len(targets)):
            error = targets[i] - output[i]

            if self.activation_type == "relu":
                derivative = self._sigmoid_derivative(output[i])
            else:
                derivative = self.activation_derivative(output[i])

            deltas_layer.append(error * derivative)
        deltas.append(deltas_layer)

        return deltas

    def _append_hidden_deltas(self, deltas: deque) -> None:
        """
        Calculate the deltas from the end to the beginning and add them to the 'deltas' list.
        """
        for row in range(len(self.weights) - 1, 0, -1):
            layer_deltas = []
            for col in range(len(self.weights[row][0])):
                error = 0
                for idx in range(len(self.weights[row])):
                    error += deltas[0][idx] * self.weights[row][idx][col]
                layer_deltas.append(
                    error * self.activation_derivative(self.layer_outputs[row][col])
                )
            deltas.appendleft(layer_deltas)

    def _update_weights(self, deltas: deque) -> None:
        """
        Updates the weights and biases according to the delta values.
        """
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += (
                        self.lr * deltas[i][j] * self.layer_outputs[i][k]
                    )
                self.biases[i][j] += self.lr * deltas[i][j]

    def _train_step(self, inputs, targets):
        output = self._forward(inputs)

        deltas = self.backpropagate(targets, output)
        self._update_weights(deltas)

        return sum(abs(targets[i] - output[i]) for i in range(len(targets))) / len(
            targets
        )

    def train(self, train_dataset, epochs=10000):
        print("Training...")
        prev_error = 0
        for n in range(epochs):
            total_error = 0
            for inp, tar in train_dataset:
                total_error += self._train_step(inp, tar)

            avg_error = total_error / len(train_dataset)

            if n % 100 == 0:
                print(f"Epoch {n}: Error = {avg_error:.6f}")

            if abs(avg_error - prev_error) < 0.000001 and n > 10:
                print(f"Stopped at Epoch {n}: Error = {avg_error:.6f}")
                break
            prev_error = avg_error


class Helper:
    def __init__(self):
        self.best_score = 0
        self.best_config = []
        self.best_activation_type = ""

    @staticmethod
    def get_file_data():
        with open("letter_to_num.json") as f:
            raw_ltn = json.load(f)
            letter_to_num = {k: tuple(v) for k, v in raw_ltn.items()}
            num_to_letter = {v: k for k, v in letter_to_num.items()}

        with open("train.json") as f:
            letters_train = json.load(f)

        return num_to_letter, letter_to_num, letters_train

    @staticmethod
    def get_dataset_from_input(letters_pattern, letter_to_num):
        return [(letters_pattern[key], letter_to_num[key]) for key in letters_pattern]

    def print_results(self, nn, test_dataset, num_to_letter, config):
        print("\n--- Test Results ---")
        correct = 0
        for inp, tar in test_dataset:
            res = nn.predict_step(inp)

            pred_char = num_to_letter.get(res)
            target_char = num_to_letter.get(tar)

            status = GREEN + "MATCH" if res == tar else RED + "FAIL"
            print(
                f"\nTarget: {target_char} | Pred: {pred_char} [{status}{RESET}] | H_config: {config} | Activation type: {nn.activation_type}"
            )

            if res == tar:
                correct += 1

            print("Input:")
            for i in range(0, 36, 6):
                row = inp[i : i + 6]
                print(
                    " ".join((GREEN if val == 1 else GREY) + str(val) for val in row)
                    + RESET
                )

        print(f"Accuracy: {correct}/{len(test_dataset)}")
        if correct > self.best_score:
            self.best_score = correct
            self.best_config = config
            self.best_activation_type = nn.activation_type

    @staticmethod
    def add_noise(train_dataset, noise_level=0.1):
        noisy_set = []
        for inp, tar in train_dataset:
            noisy_inp = [
                (1 - val if random.random() < noise_level else val) for val in inp
            ]
            noisy_set.append((noisy_inp, tar))
        return noisy_set


def main():
    helper = Helper()

    num_to_letter, letter_to_num, letters_train = Helper.get_file_data()
    dataset = helper.get_dataset_from_input(letters_train, letter_to_num)
    test_dataset = helper.add_noise(dataset, noise_level=TEST_NOISE_LEVEL)

    for config in LAYERS_CONFIG:
        for activation_type in ACTIVATIONS_TYPE:
            print(f"\nTesting Architecture: {config}")
            nn = NeuralNetwork(
                input_size=36,
                hidden_layers=config,
                output_size=2,
                activation_type=activation_type,
            )

            train_dataset = helper.add_noise(dataset, noise_level=TRAIN_NOISE_LEVEL)
            nn.train(train_dataset)

            helper.print_results(nn, test_dataset, num_to_letter, config)
            print(
                f"\n\nBEST CONFIG: {helper.best_config} with activation type: '{helper.best_activation_type}' with score: {helper.best_score}"
            )


if __name__ == "__main__":
    main()
