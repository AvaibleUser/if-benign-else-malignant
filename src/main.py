import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import StandardScaler

from controller.trainer import test, train

if __name__ == "__main__":
    data, y = load_breast_cancer(return_X_y=True, as_frame=True)
    data["y"] = y
    data = np.array(data)
    np.random.shuffle(data)

    testset = data[:200].T
    trainset = data[200:].T

    X_test = testset[:-1]
    Y_test = testset[-1]
    X_train = trainset[:-1]
    Y_train = trainset[-1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T

    neural_network = train(X_train, Y_train, 0.0005, 200)

    accuracy = test(neural_network, X_test, Y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")
