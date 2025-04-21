import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from view.colors import PINK, SKY


class DataVisualizationFrame(tk.Toplevel):
    tags = {
        "Radio medio": "mean radius",
        "Textura media": "mean texture",
        "Perímetro medio": "mean perimeter",
        "Área media": "mean area",
        "Suavidad media": "mean smoothness",
        "Compactación media": "mean compactness",
        "Concavidad media": "mean concavity",
        "Puntos concavos medios": "mean concave points",
        "Simetría media": "mean symmetry",
        "Dimensiones fractales medias": "mean fractal dimension",
        "Error de radio": "radius error",
        "Error de textura": "texture error",
        "Error de perímetro": "perimeter error",
        "Error de área": "area error",
        "Error de suavidad": "smoothness error",
        "Error de compactación": "compactness error",
        "Error de concavidad": "concavity error",
        "Error de puntos concavos": "concave points error",
        "Error de simetría": "symmetry error",
        "Error de dimensión fractal": "fractal dimension error",
        "Peor radio": "worst radius",
        "Peor textura": "worst texture",
        "Peor perímetro": "worst perimeter",
        "Peor área": "worst area",
        "Peor suavidad": "worst smoothness",
        "Peor compactación": "worst compactness",
        "Peor concavidad": "worst concavity",
        "Peores puntos concavos": "worst concave points",
        "Peor simetría": "worst symmetry",
        "Peor dimensión fractal": "worst fractal dimension",
    }

    def __init__(self, master, data, feature_names: list[str]):
        super().__init__(master)
        self.title("Visualización de Datos")

        self.data = data
        self.feature_names = feature_names

        self.setup_window()

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

    def setup_window(self):
        frame = ttk.Frame(self)
        frame.pack()

        ttk.Label(frame, text="Visualización de Datos", font=(
            "MonaspiceKr Nerd Font Mono", 16)).grid(row=0, column=0, columnspan=2, pady=20)

        figure = plt.figure(figsize=(5, 4))
        self.ax = figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(figure, master=frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.feature_x = ttk.Combobox(
            frame, values=list(self.tags.keys()), state="readonly")
        self.feature_y = ttk.Combobox(
            frame, values=list(self.tags.keys()), state="readonly")

        ttk.Label(frame, text="Característica eje X:").grid(
            row=2, column=0, padx=5, pady=2, sticky="e")
        self.feature_x.grid(row=2, column=1, padx=5, pady=10, sticky="ew")
        ttk.Label(frame, text="Característica eje Y:").grid(
            row=3, column=0, padx=5, pady=2, sticky="e")
        self.feature_y.grid(row=3, column=1, padx=5, pady=10, sticky="ew")

        ttk.Button(frame, text="Update Plot", command=self._update_plot).grid(
            row=4, column=0, columnspan=2, pady=10)

    def _update_plot(self):
        try:
            x_feat = self.feature_x.get()
            y_feat = self.feature_y.get()
            if not x_feat or not y_feat:
                raise ValueError("Debes seleccionar ambas características")

            x_idx = self.feature_names.index(self.tags.get(x_feat))
            y_idx = self.feature_names.index(self.tags.get(y_feat))
            X = self.data.data
            Y = self.data.target

            self.ax.clear()
            self.ax.scatter(
                X[Y == 0, x_idx],
                X[Y == 0, y_idx],
                c=PINK,
            )
            self.ax.scatter(
                X[Y == 1, x_idx],
                X[Y == 1, y_idx],
                c=SKY
            )
            self.ax.set_xlabel(x_feat)
            self.ax.set_ylabel(y_feat)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error de visualización", str(e))
