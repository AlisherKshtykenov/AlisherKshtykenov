# path: src/activations/output/sigmoid.py
# status: in_progress
# description:

# src/activations/output/sigmoid.py


import numpy as np

def sigmoid(z):
    """Переводит любое число в вероятность от 0 до 1."""    # docString - строка документации
    z = np.asarray(z, dtype=float)      # asarray (как массив) - превращает вход в массив NumPy
    z = np.clip(z, -500, 500)    # clip (обрезать) - ограничиваем z, иначе exp(1000) переполнится
    return 1 / (1 + np.exp(-z))         # сама формула сигмоида


if __name__ == "__main__":
    print(sigmoid(0))                   # 0.5         — ноль баллов = монетка
    print(sigmoid(10))                  # ~0.99995    — много баллов = почти точно «да»
    print(sigmoid(-10))                 # ~0.00005    — мало баллов = почти точно «нет»
    print(sigmoid([-1000, 0, 1000]))    # [0. 0.5 1.] — без предупреждений, clip сработал