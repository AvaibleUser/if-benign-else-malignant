import tkinter as tk
from queue import Queue
from threading import Thread
from tkinter import messagebox, ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from controller.trainer import test, train
from domain.dto.algorithm_config import AlgorithmConfig
from view.colors import PINK, SKY, TEAL, WHITE


class ResultsDisplayFrame(tk.Toplevel):
    def __init__(self, master, config: AlgorithmConfig):
        super().__init__(master)
        self.title("Resultados")
        self.geometry("850x650")

        self.config = config

        self.setup_window()
        self.pack_propagate(False)
        self.queue = Queue()

    def setup_window(self):
        frame = ttk.Frame(self, padding=20)
        frame.pack()

        ttk.Label(frame,
                  text="Resultados de la Red Neuronal",
                  font=("MonaspiceKr Nerd Font Mono", 16)).pack(pady=20)

        figure = plt.figure(figsize=(8, 4))
        self.loss_ax = figure.add_subplot(121)
        self.decision_ax = figure.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(figure, master=frame)
        self.canvas.get_tk_widget().pack(pady=20)

        self.accuracy_label = ttk.Label(
            frame, text="Precisión: -", font=("Helvetica", 10, "bold"))
        self.accuracy_label.pack(pady=10)

        def train_thread(): return Thread(target=self.train).start()

        ttk.Button(frame, text="Entrenar Red Neuronal",
                   command=train_thread).pack(side=tk.BOTTOM, pady=10)

    def train(self):
        self.loss_history = []

        data, y = load_breast_cancer(return_X_y=True, as_frame=True)
        data["y"] = y
        data = np.array(data)
        np.random.shuffle(data)

        test_size = int(self.config.train_size * data.shape[1])

        testset = data[:test_size].T
        trainset = data[test_size:].T

        X_test = testset[:-1]
        Y_test = testset[-1]
        X_train = trainset[:-1]
        Y_train = trainset[-1]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.T).T
        X_test = scaler.transform(X_test.T).T

        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(X_train.T)
        self.Y = Y_train.T
        self.X_min, self.X_max = self.X_pca[:, 0].min(
        ) - 1, self.X_pca[:, 0].max() + 1
        self.Y_min, self.Y_max = self.X_pca[:, 1].min(
        ) - 1, self.X_pca[:, 1].max() + 1
        self.XX, self.YY = np.meshgrid(np.arange(self.X_min, self.X_max, 0.1),
                                       np.arange(self.Y_min, self.Y_max, 0.1))
        inverse_pca = pca.inverse_transform(
            np.c_[self.XX.ravel(), self.YY.ravel()])

        Thread(target=self.look_uptdates).start()

        neural_network = train(X_train, Y_train, self.config.learning_rate,
                               self.config.epochs, inverse_pca, self.add_update)
        accuracy = test(neural_network, X_test, Y_test)

        self.accuracy_label["text"] = f"Precisión en el conjunto de prueba: {accuracy:.2%}"

        messagebox.showinfo("Entrenamiento Final",
                            f"El algoritmo ha sido entrenado, la precisión con el conjunto de prueba es de {accuracy:.2%}")

    def look_uptdates(self):
        try:
            while True:
                update = self.queue.get(block=True, timeout=5)
                self.update_results(*update)
        except Exception as e:
            print(e)

    def add_update(self, loss, accuracy, prediction):
        self.queue.put((loss, accuracy, prediction))

    def update_results(self, loss, accuracy, prediction):
        self.loss_history.append(loss)
        self.loss_ax.clear()
        self.loss_ax.plot(self.loss_history, color=TEAL)
        self.loss_ax.set_title("Error en el conjunto de entrenamiento")
        self.loss_ax.set_xlabel("Épocas")
        self.loss_ax.set_ylabel("Error")

        prediction = prediction.reshape(self.XX.shape)
        self.decision_ax.clear()
        self.decision_ax.contourf(
            self.XX, self.YY, prediction, alpha=0.5, colors=[PINK, WHITE, WHITE, SKY])
        self.decision_ax.scatter(
            self.X_pca[self.Y == 0, 0], self.X_pca[self.Y == 0, 1], c=PINK, edgecolors="k", marker="o", s=50)
        self.decision_ax.scatter(
            self.X_pca[self.Y == 1, 0], self.X_pca[self.Y == 1, 1], c=SKY, edgecolors="k", marker="o", s=50)
        self.decision_ax.set_title("Frontera de decision")

        self.accuracy_label.config(
            text=f"Precisión en el conjunto de entrenamiento: {accuracy:.2%}")
        self.canvas.draw()
