import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

#Definir la semilla seed para los algoritmos y que se puedan obtener los mismos resultados para el profe


# --- Definición de Patrones Iniciales ---
#
GLIDER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])
GOSPER_GUN = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])
R_PENTOMINO = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 0]
])

BLINKER = np.array([
    [1, 1, 1]
])

TOAD = np.array([
    [0, 1, 1, 1],
    [1, 1, 1, 0]
])


class GameOfLife:
    """
    Implementa el simulador del Juego de la Vida de Conway.
    Cada celda es un agente con reglas locales.
    """

    def __init__(self, N=40, M=40, boundary_mode='dead'):
        """
        Inicializa el mundo.
        :param N, M: Tamaño configurable del mundo (NxM) [cite: 16]
        :param boundary_mode: 'dead' (muros) o 'toroidal' (toro) [cite: 7, 16]
        """
        self.N = N
        self.M = M
        self.grid = np.zeros((N, M), dtype=int)

        # Configuración de bordes [cite: 7]
        if boundary_mode == 'toroidal':
            self.boundary_config = {'mode': 'same', 'boundary': 'wrap'}
        else:  # 'dead' por defecto
            self.boundary_config = {'mode': 'same', 'boundary': 'fill', 'fillvalue': 0}

        # Kernel para "percepción" de vecinos (Vecindad de Moore) [cite: 6]
        # El 0 central es para no contarse a sí mismo.
        self.kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

        # Para métricas [cite: 17]
        self.history = []
        self.live_agents_count = []

    def plant_seed(self, pattern, pos=(0, 0)):
        """ Coloca un patrón (semilla) en la cuadrícula. """
        r, c = pos
        rows, cols = pattern.shape
        self.grid[r:r + rows, c:c + cols] = pattern

    def plant_random(self, density=0.2, seed=None):
        """
        Genera un patrón aleatorio con una densidad dada.
        :param seed: Fija la semilla para reproducibilidad [cite: 4, 17]
        """
        # Usamos un generador de números aleatorios con semilla
        rng = np.random.default_rng(seed)
        self.grid = (rng.random((self.N, self.M)) < density).astype(int)

    def step(self):
        """
        Ejecuta un paso (ciclo) de la simulación.
        Aplica la lógica del Agente [cite: 8] de forma síncrona[cite: 4].
        """

        # 1. Percepción (Agente)
        # Contamos las vecinas vivas en t para TODAS las celdas a la vez.
        # convolve2d aplica el kernel (vecindad) a la cuadrícula actual.
        # Esto maneja los bordes (muros o toro) automáticamente.
        neighbor_count = convolve2d(self.grid, self.kernel, **self.boundary_config)

        # 2. Decisión (Agente)  y 3. Acción (Agente) [cite: 9]
        # Aplicamos las reglas B3/S23 [cite: 4, 6]

        # Regla B3: Nace con 3 vecinas
        # Solo aplica a celdas muertas (self.grid == 0)
        birth = (neighbor_count == 3) & (self.grid == 0)

        # Regla S23: Sobrevive con 2 o 3 vecinas
        # Solo aplica a celdas vivas (self.grid == 1)
        survive = ((neighbor_count == 2) | (neighbor_count == 3)) & (self.grid == 1)

        # Actualización Sincrónica
        # El nuevo estado (t+1) es la unión de las que nacen y las que sobreviven.
        # Todas las demás mueren (se vuelven 0).
        self.grid = (birth | survive).astype(int)

        # Registrar métricas [cite: 17]
        self.live_agents_count.append(np.sum(self.grid))
        self.history.append(self.grid.copy())  # Guardamos estado para análisis

        return self.grid

    def run(self, cycles=500):
        """ Ejecuta la simulación por un número dado de ciclos[cite: 16]. """
        print(f"Ejecutando simulación por {cycles} ciclos...")
        for i in range(cycles):
            self.step()
        print("Simulación completada.")

    def plot_metrics(self):
        """ Genera la gráfica de agentes vivos por paso[cite: 17, 38]. """
        if not self.live_agents_count:
            print("Ejecute la simulación (run()) primero.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.live_agents_count)
        plt.title(f'Agentes Vivos por Ciclo (Mundo {self.N}x{self.M})')
        plt.xlabel('Ciclo (Paso t)')
        plt.ylabel('Número de Agentes Vivos')
        plt.grid(True)
        plt.show()

    def animate(self, cycles=100, interval=50):
        """ Muestra una animación de la simulación. """

        # Ejecuta la simulación si no se ha hecho
        if not self.history:
            self.run(cycles)

        fig, ax = plt.subplots()
        # Usamos 'spy' para una visualización rápida de 0s y 1s
        img = ax.spy(self.history[0])

        def update(frame):
            img.set_data(self.history[frame])
            ax.set_title(f'Ciclo: {frame}')
            return img,

        # Creamos la animación
        ani = FuncAnimation(fig, update, frames=len(self.history),
                            interval=interval, blit=True, repeat=False)
        plt.show()
