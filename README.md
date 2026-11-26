# Redes Neuronales Informadas por la Física (PINNs) para ecuaciones diferenciales

Solución de ecuaciones diferenciales ordinarias y en derivadas parciales con **redes neuronales informadas por la física** (PINNs): problemas directos e inversos, implementados en Python con PyTorch y DeepXDE.

**Autor:** Lenin Herrera Padilla · lenin.herrera.p@uni.pe

## Contenido

| Archivo | Descripción | Abrir |
|---|---|---|
| [`notebooks/PINNs_EDO.ipynb`](notebooks/PINNs_EDO.ipynb) | Ecuaciones diferenciales ordinarias: red estándar frente a PINN, elección de la función de activación, problemas directos e inversos, problema rígido, problema de contorno y límites del método | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ldherrerap/pinns-ecuaciones-diferenciales/blob/main/notebooks/PINNs_EDO.ipynb) |
| [`notebooks/PINNs_EDP.ipynb`](notebooks/PINNs_EDP.ipynb) | Ecuaciones en derivadas parciales: ecuación del calor (directo e inverso) y ecuación de onda en 1D y 2D | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ldherrerap/pinns-ecuaciones-diferenciales/blob/main/notebooks/PINNs_EDP.ipynb) |

## La idea

Una PINN aproxima la solución por una red neuronal $u_\theta(t)$ y la entrena para que cumpla la ecuación. Las derivadas de la red se calculan por diferenciación automática y el residuo de la ecuación entra en la función de pérdida:

$$
r_\theta(t)=u_\theta'(t)-f\big(t,u_\theta(t)\big),\qquad
\mathcal{L}(\theta)=\frac{1}{N_d}\sum_{i=1}^{N_d}\big|u_\theta(t_i)-u_i\big|^2+\frac{1}{N_c}\sum_{j=1}^{N_c}\big|r_\theta(t_j)\big|^2 .
$$

En un **problema inverso**, los parámetros desconocidos de la ecuación se declaran entrenables y se estiman junto con los pesos de la red, en un solo entrenamiento.

## Casos de estudio

### Ecuaciones diferenciales ordinarias (`PINNs_EDO.ipynb`)

| Caso | Tipo | Arquitectura | Resultado |
|---|---|---|---|
| Oscilador amortiguado con 10 datos | red estándar frente a PINN | 1–32–32–32–1, tanh | la PINN extrapola donde no hay datos |
| Oscilador de alta frecuencia | tanh frente a seno (SIREN) | 1–64–64–64–64–1 | el seno resuelve lo que tanh no |
| Ecuación logística | directo | 1–32–32–32–1, tanh | error L² 3,6 × 10⁻⁵ |
| Oscilador armónico simple | directo (ecuación y sistema) | 1–32–32–32–1, tanh | mismo orden de error en ambas formulaciones |
| Propagación de una gripe | inverso | 1–32–32–32–1, tanh | r = 0,9906 |
| Oscilador con datos de un simulador | inverso | 1–64–64–64–1, tanh | k con error de 0,20 % |
| Sistema de Lorenz (ρ = 15) | inverso | 1–40–40–40–3, tanh | σ, ρ, β con error menor a 0,5 % |
| Problema rígido | directo | 1–64–64–64–64–1, tanh | estable donde RK4 explícito diverge |
| Problema de contorno | directo | 1–32–32–32–1, tanh | error L² 1,3 × 10⁻⁴, sin método de disparo |
| Lorenz caótico y Van der Pol | límites del método | — | no convergen |

### Ecuaciones en derivadas parciales (`PINNs_EDP.ipynb`)

| Caso | Tipo | Arquitectura | Resultado |
|---|---|---|---|
| Ecuación del calor 1D | directo | 2–32–32–32–1, tanh | error L² 3,1 × 10⁻⁴ |
| Ecuación del calor 1D | inverso (difusividad α) | 2–32–32–32–1, tanh | α = 0,09991 (error 0,09 %) |
| Cuerda vibrante | directo | 2–64–64–64–64–1, tanh | error L² 4,0 × 10⁻⁵ |
| Onda 2D con velocidad variable | directo, referencia por diferencias finitas | 3–64–64–64–64–1, tanh | error L² de 1,3 × 10⁻² a 1,4 × 10⁻¹ |

## Uso

Los notebooks están preparados para **Google Colab** (botones de la tabla) y se ejecutan en orden. Los datos necesarios van incluidos en los propios notebooks.

- `RAPIDO = True`, en la primera celda de código, reduce unas diez veces los entrenamientos para una ejecución rápida; con `RAPIDO = False` se obtienen los resultados completos.
- La onda 2D es el caso más costoso: en Colab conviene activar la GPU (*Entorno de ejecución → Cambiar tipo de entorno*).

Para ejecutarlos localmente:

```bash
pip install torch deepxde scipy numpy pandas matplotlib
```

## Referencias

- Raissi, M., Perdikaris, P. y Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Lu, L., Meng, X., Mao, Z. y Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.
- Moseley, B. (2022). *So, what is a physics-informed neural network?* La comparación entre red estándar y PINN del oscilador amortiguado es una adaptación de su código.
- Moseley, B., Markham, A. y Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs).
- Sitzmann, V. et al. (2020). Implicit neural representations with periodic activation functions (SIREN).
- Zill, D. G. (2009). *Ecuaciones diferenciales con aplicaciones de modelado*.

## Licencia

El código se distribuye bajo la licencia [MIT](LICENSE).
