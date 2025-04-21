import tkinter as tk
from tkinter import ttk

import catppuccin
import matplotlib as mpl
from sklearn.datasets import load_breast_cancer

from domain.dto.algorithm_config import AlgorithmConfig
from view.data import DataVisualizationFrame
from view.results import ResultsDisplayFrame
from view.training import TrainingControlFrame


class Principal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.tk.call("source", "themes/azure.tcl")
        self.tk.call("set_theme", "dark")

        mpl.style.use(catppuccin.PALETTE.mocha.identifier)

        self.title("Clasificador de cáncer de mama")
        self.geometry("600x300")
        self.setup_window()

        self.config = AlgorithmConfig()

        self.raw_data = load_breast_cancer()
        self.feature_names = self.raw_data.feature_names.tolist()

    def setup_window(self):
        principal_frame = ttk.Frame(self)
        principal_frame.pack(expand=True, fill="both", padx=20, pady=20)

        ttk.Label(principal_frame,
                  text="Clasificador de cáncer de mama",
                  font=("MonaspiceKr Nerd Font Mono", 16)).pack(pady=20)

        buttons = [
            ("Visualizar Datos", self.open_dataset_visualization),
            ("Configurar Red Neuronal", self.open_algorithm_config),
            ("Generar Red Neuronal", self.start_generation),
            ("Salir", self.destroy)
        ]

        for text, command in buttons:
            ttk.Button(principal_frame, text=text, command=command,
                       width=30).pack(pady=5)

    def open_dataset_visualization(self):
        self.data_viz = DataVisualizationFrame(
            self, self.raw_data, self.feature_names)

    def open_algorithm_config(self):
        self.training_control = TrainingControlFrame(self, self.config)

    def start_generation(self):
        self.results_display = ResultsDisplayFrame(self, self.config)
