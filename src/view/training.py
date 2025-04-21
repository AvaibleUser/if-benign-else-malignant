import tkinter as tk
from tkinter import messagebox, ttk

from domain.dto.algorithm_config import AlgorithmConfig


class TrainingControlFrame(tk.Toplevel):
    def __init__(self, master, config: AlgorithmConfig):
        super().__init__(master)
        self.title("Configuración del Algoritmo")

        self.config = config

        self.setup_window()
        self.grid_columnconfigure(1, weight=1)

    def setup_window(self):
        frame = ttk.Frame(self, padding=20)
        frame.pack()

        ttk.Label(frame,
                  text="Configuración de la Red Neuronal",
                  font=("MonaspiceKr Nerd Font Mono", 16)).grid(row=0, column=0, columnspan=2, pady=20)

        self.lr_var = tk.DoubleVar(value=self.config.learning_rate)
        self.epochs_var = tk.IntVar(value=self.config.epochs)
        self.train_size_var = tk.DoubleVar(value=self.config.train_size)

        ttk.Label(frame, text="Ratio de Aprendizaje (η):").grid(
            row=1, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame, textvariable=self.lr_var).grid(
            row=1, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(frame, text="Cantidad de Épocas:").grid(
            row=2, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame, textvariable=self.epochs_var).grid(
            row=2, column=1, padx=5, pady=2, sticky="ew")

        def update_train_size(v):
            self.train_size_var.set(round(float(v), 2))
            self.train_size_label["text"] = f"Valor actual: {self.train_size_var.get():.2%}"

        ttk.Label(frame, text="Tamaño de la Muestra de Entrenamiento:").grid(
            row=3, column=0, padx=5, pady=2, sticky="e")
        ttk.Scale(frame, from_=0.5, to=0.95, variable=self.train_size_var,
                  command=update_train_size).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        self.train_size_label = ttk.Label(
            frame, text=f"Valor actual: {self.train_size_var.get():.2%}", font=("MonaspiceRn Nerd Font Mono", 8))
        self.train_size_label.grid(row=4, column=1, padx=5, pady=2, sticky="e")

        ttk.Button(frame, text="Guardar Configuración", command=self.save_configuration).grid(
            row=5, column=0, columnspan=2, pady=10)

    def save_configuration(self):
        try:
            lr = self.lr_var.get()
            if lr <= 0 or lr > 1:
                raise ValueError(
                    "La tasa de aprendizaje debe estar entre 0 y 1")

            epochs = self.epochs_var.get()
            if epochs < 10 or epochs > 1000:
                raise ValueError("Número de épocas inválido (10-1000)")

            self.config.learning_rate = lr
            self.config.epochs = epochs
            self.config.train_size = self.train_size_var.get()

            messagebox.showinfo("Configuración Guardada",
                                "La configuración ha sido guardada")
            self.destroy()

        except Exception as e:
            messagebox.showerror("Error de validación",
                                 f"Parámetros incorrectos:\n{str(e)}")
