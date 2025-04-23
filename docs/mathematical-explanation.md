# Explicación Matemática del Modelo de Red Neuronal

## Estructura de la Red Neuronal
- **Arquitectura**: Perceptrón multicapa (MLP) con una capa oculta.
  - **Capa de entrada**: 30 neuronas, correspondientes a todas las características del dataset.
  - **Capa oculta**: $m$ neuronas con función de activación _ReLU_.
  - **Capa de salida**: 1 neurona con función de activación _sigmoide_ para clasificación binaria.

## Funciones Clave
1. **Función de Activación**:
   - **Sigmoide** (capa de salida):
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
   - **ReLU** (capa oculta):
$$
\text{ReLU}(z) = \max(0, z)
$$

2. **Función de Pérdida** (Entropía Cruzada Binaria):
$$
\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
   - $y$: Etiqueta real (0 o 1).
   - $\hat{y}$: Predicción de la red.

## Inicialización de pesos y sesgos
- **Inicialización de pesos**:
	- **Inicialización He (o Kaiming)**:
$$
\begin{equation}
\mathbf{W}_{L} = \mathbf{N} \sqrt{\frac{2}{n_{L-1}}}

\quad\text{donde:}\quad

\begin{split}
\mathbf{N} &= \text{Numero aleatorio} \\ \\
n_{L-1} &= \text{Cantidad de entradas}
\end{split}
\end{equation}
$$
- **Inicialización de sesgos**:
	- **Inicialización en ceros**:
$$
\mathbf{b}_{L} = 0
$$

## Propagación hacia Adelante (Forward Pass)
- **Salida de la capa oculta**:
$$
\mathbf{A}_{1} = \text{ReLU}(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)
$$
  - $\text{ReLU}(\dots)$: Función de activación, ReLU
  - $\mathbf{X}$: Matriz de entrada con 30 características.
  - $\mathbf{W}_1$: Matriz de pesos de la capa.
  - $\mathbf{b}_1$: Vector de sesgos

- **Salida de la red**:
$$
\hat{y} = \sigma(\mathbf{A}_{1} \mathbf{W}_2 + \mathbf{b}_2)
$$
  - $\sigma(\dots)$: Función de activación, sigmoide
  - $\mathbf{A}_{1}$: Matriz de entrada con la cantidad de salidas de la capa anterior.
  - $\mathbf{W}_2$: Matriz de pesos de la capa de salida.
  - $\mathbf{b}_2$: Vector de sesgos

## Retropropagación (Backpropagation/Backward Pass)
1. **Cálculo de Gradientes de la capa de salida**:

   - **Error imputado a la capa de salida $\delta^2$:
$$
\begin{equation}

\begin{split}
\delta^2 = \frac{\partial \mathcal{L}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \mathbf{z}_{2}}
\end{split}

\quad\text{donde:}\quad

\begin{split}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= - \frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} \\ \\

\frac{\partial \hat{y}}{\partial \mathbf{z}_{2}} &= \sigma(z_{2})\times(1 - \sigma(z_{2}))
\end{split}

\end{equation}
$$
   - **Gradiente del costo respecto a $\mathbf{W}_2$**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \delta^2 \frac{\partial \mathbf{z}_{2}}{\partial \mathbf{W}_{2}}

\quad\text{donde:}\quad

\frac{\partial \mathbf{z}_{2}}{\partial \mathbf{W}_{2}} = \mathbf{A}_{1}

\quad\Rightarrow\quad

\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \delta^2 \mathbf{A}_{1}
$$
   - **Gradiente respecto a $\mathbf{b}_2$**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \delta^2 \frac{\partial \mathbf{z}_{2}}{\partial \mathbf{b}_{2}}

\quad\text{donde:}\quad

\frac{\partial \mathbf{z}_{2}}{\partial \mathbf{b}_{2}} = 1

\quad\Rightarrow\quad

\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \delta^2
$$
2. **Cálculo de Gradientes de la capa oculta**:

   - **Error imputado a la capa oculta $\delta^1$:
$$
\begin{equation}

\begin{split}
\delta^1 = \delta^2 \frac{\partial \mathbf{z}_2}{\partial \mathbf{A}_{1}} \frac{\partial \mathbf{A}_{1}}{\partial \mathbf{z}_1}
\end{split}

\quad\text{donde:}\quad

\begin{split}
\frac{\partial \mathbf{z}_2}{\partial \mathbf{A}_{1}} &= \mathbf{W}_{2} \\ \\

\frac{\partial \hat{y}}{\partial \mathbf{z}} &= \sigma(z)\times(1 - \sigma(z))
\end{split}

\end{equation}
$$
   - **Gradiente del costo respecto a $\mathbf{W}_2$**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta^1 \frac{\partial \mathbf{z}_{1}}{\partial \mathbf{W}_{1}}

\quad\text{donde:}\quad

\frac{\partial \mathbf{z}_{1}}{\partial \mathbf{W}_{1}} = \mathbf{X}

\quad\Rightarrow\quad

\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta^1 \mathbf{X}
$$
   - **Gradiente respecto a $\mathbf{b}_2$**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta^1 \frac{\partial \mathbf{z}_{1}}{\partial \mathbf{b}_{1}}

\quad\text{donde:}\quad

\frac{\partial \mathbf{z}_{1}}{\partial \mathbf{b}_{1}} = 1

\quad\Rightarrow\quad

\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta^1
$$

2. **Actualización de Pesos**:
$$
\mathbf{W}_{L} \leftarrow \mathbf{W}_{L} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{L}}
$$
3. **Actualización de Sesgos:
$$
\mathbf{b}_{L} \leftarrow \mathbf{b}_{L} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}_{L}}
$$
## Ajustes para la Visualización 2D
Aunque el modelo utiliza 30 características** para entrenar, la interfaz gráfica proyecta los datos en 2D mediante:
- Selección manual de dos características en la visualización de datos.
- La frontera de decisión mostrada es una proyección de la superficie no lineal de las 30 características condensadas en 2 características mediante el uso del _Análisis de componentes principales_ o PCA.
