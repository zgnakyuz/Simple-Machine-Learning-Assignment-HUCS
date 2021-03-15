# coding=utf-8
# Özgün Akyüz, 12.01.2020
# Single Layer Neural Network
# Python 2.7
import numpy as np  # 1.16.4
import pandas as pd  # 0.24.2
import matplotlib.pyplot as plt  # 2.2.4


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def read_and_divide_into_train_and_test(csv_file):
    dataset = pd.read_csv(csv_file)
    missing_values = list()

    # Completing missing values with the mean value of that column
    for i, colName in enumerate(list(dataset.keys())[1:]):
        for j, value in enumerate(dataset[colName]):
            if value == "?":
                missing_values.append([j, i + 1])
                dataset.iat[j, i + 1] = 0

    dataset = dataset.astype(float)
    for j, i in missing_values:
        avg = dataset[dataset.keys()[i]].mean()
        dataset.iat[j, i] = avg

    attributes = dataset.keys()[1:-1]  # Features
    len_attrs = len(attributes)

    # Requested data
    training_set = dataset.head(559)
    training_inputs = np.array(training_set.drop(["Code_number", "Class"], axis=1))
    training_labels = np.array(training_set["Class"])
    training_labels = training_labels.reshape(training_labels.shape[0], -1)

    test_set = dataset.tail(140)
    test_inputs = np.array(test_set.drop(["Code_number", "Class"], axis=1))
    test_labels = np.array(test_set["Class"])

    # Visualization of pairwise correlations of the attributes of training set
    corr_array = np.array(training_set.drop(["Code_number", "Class"], axis=1).corr())

    fig, axis = plt.subplots()
    fig.canvas.set_window_title("Correlations")
    axis.set_title("Pairwise correlations of the attributes of training set")
    im = axis.imshow(corr_array)

    axis.set_xticks(np.arange(len_attrs)), axis.set_yticks(np.arange(len_attrs))
    axis.set_xticklabels(attributes), axis.set_yticklabels(attributes)
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    for i in range(len_attrs):
        for j in range(len_attrs):
            axis.text(j, i, round(corr_array[i, j], 2), ha="center", va="center", color="r")

    fig.colorbar(im)
    fig.subplots_adjust(bottom=0.27, right=0.788)
    plt.show()

    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    total_number_of_samples = float(len(test_inputs))

    test_predictions = sigmoid(np.dot(test_inputs, weights))

    for i, line in enumerate(test_predictions):  # Converting value to 1 or 0
        for j, x in enumerate(line):
            if x > 0.5:
                test_predictions[i][j] = 1
            else:
                test_predictions[i][j] = 0

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1

    accuracy = tp / total_number_of_samples
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    fig = plt.figure()
    fig.canvas.set_window_title("Changes")
    plt1 = fig.add_subplot(211)
    plt2 = fig.add_subplot(212)

    plt1.set_title("Training Loss", fontsize=15)
    plt1.plot(loss_array, "#0909FF")
    plt1.set_ylabel("Loss", fontsize=13)

    plt2.set_title("Test Accuracy", fontsize=15)
    plt2.plot(accuracy_array, "#058205")
    plt2.set_xlabel("# Epochs", fontsize=13)
    plt2.set_ylabel("Accuracy", fontsize=13)

    fig.subplots_adjust(bottom=0.1)
    plt.show()


def main():
    csv_file = "./breast-cancer-wisconsin.csv"
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        # Forward Propagation
        outputs = np.dot(training_inputs, weights)
        outputs = sigmoid(outputs)

        # Backward Propagation
        loss = training_labels - outputs
        tuning = loss * sigmoid_derivative(outputs)

        weights += np.dot(np.transpose(training_inputs), tuning)

        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
        loss_array.append(np.mean(loss))

    plot_loss_accuracy(accuracy_array, loss_array)


if __name__ == '__main__':
    main()
